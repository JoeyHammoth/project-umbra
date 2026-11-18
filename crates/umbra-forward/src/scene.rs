//! The scene document (spec §8.1): loader, validation, identity and density.
//!
//! A scene is the one artefact every downstream task consumes, so the rules here
//! are deliberately strict:
//!
//! - **Unknown keys are rejected**, not ignored. Spec §8.1 makes this binding,
//!   and the reason is that silently dropping a field is how two people get
//!   different physics from the same file while both believing they loaded it.
//! - **Units are stated and checked.** A scene that says `g/cm3` when it means
//!   `kg/m3` is a factor of a thousand in every count, so the loader refuses any
//!   unit string but the one it actually applies.
//! - **Material must lie inside the grid.** A primitive that pokes outside the
//!   voxel grid is material the forward model cannot see: the ray integrals stop
//!   at the grid edge, so the scene would under-report its own opacity. That is
//!   rejected rather than warned about.
//!
//! # Identity
//!
//! [`SceneSpec::scene_key_hex`] hashes the *physics-relevant* fields only:
//! geometry, densities, detectors, physics parameters, exposure, seed and grid.
//! Labels, prose and notes are excluded on purpose, because editing a
//! description must not invalidate every cached result in the app — while
//! changing a density by one digit must.
//!
//! # Density
//!
//! [`SceneSpec::density`] samples each solid at the **voxel centre**, adds a
//! solid's `rho` where it contains the centre and subtracts the bulk density for
//! each void, clamping at zero. That is the same centre-sampling rule the Python
//! oracle's `rasterise` uses, which is what makes the committed parity fixtures
//! comparable with the app rather than merely similar to it.

use std::fmt;

use umbra_core::json::{self, Json};
use umbra_core::{sha256, Grid2, Primitive, Quat};
use umbra_physics::{
    AngularBinning, Detector, Fidelity, Pose, STANDARD_ROCK_A, STANDARD_ROCK_B, STANDARD_ROCK_RHO,
    STANDARD_ROCK_X0,
};

use crate::build::Scene2D;
use crate::expected::ForwardParams;

/// The scene schema this loader understands.
pub const SCENE_SCHEMA: &str = "umbra.scene/1";

/// The golden-dataset schema written by `umbra-cli golden`.
pub const GOLDEN_SCHEMA: &str = "umbra.golden/1";

/// A solid with the density it contributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneSolid {
    /// Geometry, in metres.
    pub shape: Primitive,
    /// Density of the material, in g cm⁻³.
    pub rho_g_cm3: f32,
}

/// The physics parameters a scene carries (spec §8.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScenePhysics {
    /// Fidelity level.
    pub level: Fidelity,
    /// CSDA ionisation term, in GeV cm² g⁻¹.
    pub a: f32,
    /// CSDA radiative term, in cm² g⁻¹.
    pub b: f32,
    /// Bulk density of the medium, in g cm⁻³. Voids remove this.
    pub rho_bulk: f32,
    /// Radiation length, in cm. Carried for the analytic chord model (A7).
    pub x0: f32,
}

impl Default for ScenePhysics {
    fn default() -> Self {
        Self {
            level: Fidelity::DEFAULT,
            a: STANDARD_ROCK_A,
            b: STANDARD_ROCK_B,
            rho_bulk: STANDARD_ROCK_RHO,
            x0: STANDARD_ROCK_X0,
        }
    }
}

/// A complete scene: geometry, detectors, physics, exposure and the grid it is
/// rasterised on.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneSpec {
    /// Stable identifier, used in URLs, cache keys and golden files.
    pub id: String,
    /// Human-readable label, shown in the UI.
    pub label: String,
    /// Whether the geometry is schematic and must display the §8.3 badge.
    pub schematic: bool,
    /// Solids, in declaration order.
    pub primitives: Vec<SceneSolid>,
    /// Voids: regions where the bulk material is absent.
    pub voids: Vec<Primitive>,
    /// Detectors, in order. Order is part of the scene identity.
    pub detectors: Vec<Detector>,
    /// Physics parameters.
    pub physics: ScenePhysics,
    /// Exposure, in seconds.
    pub exposure_s: f32,
    /// Voxel grid.
    pub grid: Grid2,
    /// RNG seed for the exposure simulator.
    pub seed: u64,
    /// Free-text note carried with the scene.
    pub notes: Option<String>,
}

/// A scene or golden document that could not be loaded, with the path to the
/// field that caused it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneError {
    /// Where in the document the problem is, e.g. `scene.primitives[0].rho`.
    pub path: String,
    /// What is wrong, in a form shown straight to a user.
    pub problem: String,
}

impl SceneError {
    /// Builds an error for a field path.
    pub fn new(path: impl Into<String>, problem: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            problem: problem.into(),
        }
    }
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.problem)
    }
}

impl std::error::Error for SceneError {}

impl From<json::JsonError> for SceneError {
    fn from(error: json::JsonError) -> Self {
        SceneError::new("<document>", error.to_string())
    }
}

/// The stable lowercase id of a fidelity level, as used in documents.
pub fn fidelity_id(fidelity: Fidelity) -> &'static str {
    match fidelity {
        Fidelity::L0 => "L0",
        Fidelity::L1 => "L1",
        Fidelity::L2 => "L2",
    }
}

/// SHA-256 of a count vector, the golden file's contract.
///
/// Counts are hashed as big-endian `u32`s in ray order, so a changed count
/// anywhere — including a reordering that no aggregate would notice — changes
/// the digest.
pub fn counts_hash(counts: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(counts.len() * 4);
    for count in counts {
        bytes.extend_from_slice(&count.to_be_bytes());
    }
    umbra_core::sha256_hex(&bytes)
}

impl SceneSpec {
    /// Parses a scene document.
    pub fn from_json_str(text: &str) -> Result<Self, SceneError> {
        let value = json::parse(text)?;
        Self::from_json(&value)
    }

    /// Builds a scene from an already-parsed JSON value.
    pub fn from_json(value: &Json) -> Result<Self, SceneError> {
        let root = Reader::new("scene", value)?;
        root.reject_unknown(&[
            "schema",
            "id",
            "label",
            "schematic",
            "units",
            "primitives",
            "voids",
            "detectors",
            "physics",
            "exposure_s",
            "grid",
            "seed",
            "notes",
        ])?;

        let schema = root.string("schema")?;
        if schema != SCENE_SCHEMA {
            return Err(SceneError::new(
                "scene.schema",
                format!("expected `{SCENE_SCHEMA}`, found `{schema}`"),
            ));
        }

        // Units are checked rather than assumed: the loader applies the numbers
        // as metres and g/cm3, so any other unit string is a silent factor.
        let units = Reader::new("scene.units", root.get("units")?)?;
        units.reject_unknown(&["length", "density"])?;
        if units.string("length")? != "m" {
            return Err(SceneError::new(
                "scene.units.length",
                "only `m` is supported (the loader does not convert)",
            ));
        }
        if units.string("density")? != "g/cm3" {
            return Err(SceneError::new(
                "scene.units.density",
                "only `g/cm3` is supported (the loader does not convert)",
            ));
        }

        let primitives = root
            .array("primitives")?
            .iter()
            .enumerate()
            .map(|(index, value)| parse_solid(value, &format!("scene.primitives[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;

        let voids = match root.optional("voids") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| SceneError::new("scene.voids", "expected an array"))?
                .iter()
                .enumerate()
                .map(|(index, value)| parse_shape(value, &format!("scene.voids[{index}]")))
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };

        let detectors = root
            .array("detectors")?
            .iter()
            .enumerate()
            .map(|(index, value)| parse_detector(value, &format!("scene.detectors[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;

        let physics = match root.optional("physics") {
            Some(value) => parse_physics(value)?,
            None => ScenePhysics::default(),
        };

        let grid_reader = Reader::new("scene.grid", root.get("grid")?)?;
        grid_reader.reject_unknown(&["n", "h_m", "origin_m"])?;
        let n = grid_reader.u32_pair("n")?;
        let origin = grid_reader.vec2("origin_m")?;
        let grid = Grid2::new(n, origin, grid_reader.f32("h_m")?);

        let seed = root.u64("seed")?;
        let notes = root.optional("notes").map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| SceneError::new("scene.notes", "expected a string"))
        });
        let notes = notes.transpose()?;

        let spec = SceneSpec {
            id: root.string("id")?.to_string(),
            label: root.string("label")?.to_string(),
            schematic: match root.optional("schematic") {
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| SceneError::new("scene.schematic", "expected a boolean"))?,
                None => false,
            },
            primitives,
            voids,
            detectors,
            physics,
            exposure_s: root.f32("exposure_s")?,
            grid,
            seed,
            notes,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Checks the invariants a loadable scene must satisfy.
    pub fn validate(&self) -> Result<(), SceneError> {
        let grid = &self.grid;
        if grid.n[0] == 0 || grid.n[1] == 0 {
            return Err(SceneError::new(
                "scene.grid.n",
                "both axes need at least one cell",
            ));
        }
        if !grid.h_m.is_finite() || grid.h_m <= 0.0 || grid.h_m > 10.0 {
            return Err(SceneError::new(
                "scene.grid.h_m",
                format!("cell size {} m is outside (0, 10] m", grid.h_m),
            ));
        }
        if grid.cell_count() > 4_000_000 {
            return Err(SceneError::new(
                "scene.grid.n",
                format!(
                    "{} cells exceeds the 4,000,000-cell limit",
                    grid.cell_count()
                ),
            ));
        }
        if !self.exposure_s.is_finite() || self.exposure_s <= 0.0 {
            return Err(SceneError::new(
                "scene.exposure_s",
                "exposure must be positive",
            ));
        }
        if self.primitives.is_empty() {
            return Err(SceneError::new(
                "scene.primitives",
                "a scene needs at least one solid",
            ));
        }
        if self.detectors.is_empty() {
            return Err(SceneError::new(
                "scene.detectors",
                "a scene needs at least one detector",
            ));
        }

        for (index, solid) in self.primitives.iter().enumerate() {
            if !solid.rho_g_cm3.is_finite() || solid.rho_g_cm3 < 0.0 {
                return Err(SceneError::new(
                    format!("scene.primitives[{index}].rho"),
                    "density must be finite and non-negative",
                ));
            }
            self.check_inside_grid(&solid.shape, format!("scene.primitives[{index}]"))?;
        }
        for (index, void) in self.voids.iter().enumerate() {
            self.check_inside_grid(void, format!("scene.voids[{index}]"))?;
        }

        for (index, detector) in self.detectors.iter().enumerate() {
            let path = format!("scene.detectors[{index}]");
            if !detector.area_m2.is_finite() || detector.area_m2 < 0.0 {
                return Err(SceneError::new(
                    format!("{path}.area_m2"),
                    "area must be non-negative",
                ));
            }
            if !(0.0..=1.0).contains(&detector.eff) {
                return Err(SceneError::new(
                    format!("{path}.eff"),
                    "efficiency must be in [0, 1]",
                ));
            }
            if !detector.sigma_ang_mrad.is_finite() || detector.sigma_ang_mrad < 0.0 {
                return Err(SceneError::new(
                    format!("{path}.sigma_ang_mrad"),
                    "resolution must be non-negative",
                ));
            }
            if !detector.bg_rate_hz.is_finite() || detector.bg_rate_hz < 0.0 {
                return Err(SceneError::new(
                    format!("{path}.bg_rate_hz"),
                    "background rate must be non-negative",
                ));
            }
            match detector.binning {
                AngularBinning::Flat2D { min, max, bins } => {
                    if bins == 0 {
                        return Err(SceneError::new(
                            format!("{path}.bins"),
                            "need at least one bin",
                        ));
                    }
                    if !max.is_finite() || !min.is_finite() || max <= min {
                        return Err(SceneError::new(
                            format!("{path}.bins"),
                            "the angular range must have max > min",
                        ));
                    }
                }
                AngularBinning::AzZen {
                    az_min,
                    az_max,
                    az_bins,
                    zen_min,
                    zen_max,
                    zen_bins,
                } => {
                    if az_bins == 0 || zen_bins == 0 {
                        return Err(SceneError::new(
                            format!("{path}.bins"),
                            "need at least one bin",
                        ));
                    }
                    if !az_max.is_finite()
                        || !az_min.is_finite()
                        || !zen_max.is_finite()
                        || !zen_min.is_finite()
                        || az_max <= az_min
                        || zen_max <= zen_min
                    {
                        return Err(SceneError::new(
                            format!("{path}.bins"),
                            "both angular ranges must have max > min",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_inside_grid(&self, shape: &Primitive, path: String) -> Result<(), SceneError> {
        let (lower, upper) = shape.bbox_xy();
        let (grid_lower, grid_upper) = self.grid_bounds();
        let disjoint = upper[0] < grid_lower[0]
            || upper[1] < grid_lower[1]
            || lower[0] > grid_upper[0]
            || lower[1] > grid_upper[1];
        if disjoint {
            return Err(SceneError::new(
                path,
                format!(
                    "geometry [{}..{}] x [{}..{}] m lies outside the grid \
                     [{}..{}] x [{}..{}] m, so the forward model could not see it",
                    lower[0],
                    upper[0],
                    lower[1],
                    upper[1],
                    grid_lower[0],
                    grid_upper[0],
                    grid_lower[1],
                    grid_upper[1]
                ),
            ));
        }
        Ok(())
    }

    /// World-space bounds of the grid, as `(lower, upper)`.
    pub fn grid_bounds(&self) -> ([f32; 2], [f32; 2]) {
        let grid = &self.grid;
        (
            grid.origin,
            [
                grid.origin[0] + grid.n[0] as f32 * grid.h_m,
                grid.origin[1] + grid.n[1] as f32 * grid.h_m,
            ],
        )
    }

    /// Canonical bytes of the physics-relevant fields.
    ///
    /// Prose, labels and notes are excluded: the key is the identity of the
    /// experiment, not of its description.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_string(&mut out, SCENE_SCHEMA);
        push_string(&mut out, &self.id);

        push_u32(&mut out, self.grid.n[0]);
        push_u32(&mut out, self.grid.n[1]);
        push_f32(&mut out, self.grid.origin[0]);
        push_f32(&mut out, self.grid.origin[1]);
        push_f32(&mut out, self.grid.h_m);

        push_u32(&mut out, self.primitives.len() as u32);
        for solid in &self.primitives {
            push_primitive(&mut out, &solid.shape);
            push_f32(&mut out, solid.rho_g_cm3);
        }
        push_u32(&mut out, self.voids.len() as u32);
        for void in &self.voids {
            push_primitive(&mut out, void);
        }

        push_u32(&mut out, self.detectors.len() as u32);
        for detector in &self.detectors {
            for value in detector
                .pose
                .position
                .iter()
                .chain(detector.pose.normal.iter())
                .chain(detector.pose.up.iter())
            {
                push_f32(&mut out, *value);
            }
            push_f32(&mut out, detector.area_m2);
            push_f32(&mut out, detector.eff);
            push_f32(&mut out, detector.sigma_ang_mrad);
            push_f32(&mut out, detector.bg_rate_hz);
            match detector.binning {
                AngularBinning::Flat2D { min, max, bins } => {
                    push_u32(&mut out, 0);
                    push_f32(&mut out, min);
                    push_f32(&mut out, max);
                    push_u32(&mut out, bins);
                }
                AngularBinning::AzZen {
                    az_min,
                    az_max,
                    az_bins,
                    zen_min,
                    zen_max,
                    zen_bins,
                } => {
                    push_u32(&mut out, 1);
                    push_f32(&mut out, az_min);
                    push_f32(&mut out, az_max);
                    push_u32(&mut out, az_bins);
                    push_f32(&mut out, zen_min);
                    push_f32(&mut out, zen_max);
                    push_u32(&mut out, zen_bins);
                }
            }
        }

        push_f32(&mut out, self.physics.a);
        push_f32(&mut out, self.physics.b);
        push_f32(&mut out, self.physics.rho_bulk);
        push_f32(&mut out, self.physics.x0);
        push_string(&mut out, fidelity_id(self.physics.level));

        push_f32(&mut out, self.exposure_s);
        out.extend_from_slice(&self.seed.to_be_bytes());
        out
    }

    /// Hex SHA-256 of [`SceneSpec::canonical_bytes`].
    pub fn scene_key_hex(&self) -> String {
        umbra_core::sha256_hex(&self.canonical_bytes())
    }

    /// The scene key as a `u64`, for the RNG stream ids.
    pub fn scene_key(&self) -> u64 {
        let digest = sha256(&self.canonical_bytes());
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        u64::from_be_bytes(bytes)
    }

    /// Density on the voxel grid, in g cm⁻³.
    ///
    /// Solids add; voids remove the bulk density; the result is clamped at zero
    /// because a void carved twice is still empty, not negative.
    pub fn density(&self) -> Vec<f32> {
        let grid = &self.grid;
        let mut field = vec![0.0f32; grid.cell_count()];
        for j in 0..grid.n[1] {
            for i in 0..grid.n[0] {
                let centre = grid.centre(i, j);
                let mut value = 0.0f32;
                for solid in &self.primitives {
                    if solid.shape.contains_xy(centre) {
                        value += solid.rho_g_cm3;
                    }
                }
                for void in &self.voids {
                    if void.contains_xy(centre) {
                        value -= self.physics.rho_bulk;
                    }
                }
                if let Some(index) = grid.index(i, j) {
                    field[index] = value.max(0.0);
                }
            }
        }
        field
    }

    /// The slice-mode scene the system builder consumes.
    pub fn to_scene_2d(&self) -> Scene2D {
        Scene2D::new(self.grid.clone(), self.detectors.clone())
    }

    /// Forward-model parameters carried by this scene.
    ///
    /// `background_hz` stays zero: `expected` applies a single global rate while
    /// a scene carries one rate per detector, and inventing a combination rule
    /// here would hide that mismatch. Per-detector rates remain on the detectors
    /// for the L2 path that will consume them.
    pub fn forward_params(&self) -> ForwardParams {
        ForwardParams {
            a: self.physics.a,
            b: self.physics.b,
            ..ForwardParams::default()
        }
    }
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn push_primitive(out: &mut Vec<u8>, shape: &Primitive) {
    match shape {
        Primitive::Box { c, half, rot } => {
            push_u32(out, 0);
            for value in c.iter().chain(half.iter()).chain(rot_slice(rot).iter()) {
                push_f32(out, *value);
            }
        }
        Primitive::Ellip { c, semi, rot } => {
            push_u32(out, 1);
            for value in c.iter().chain(semi.iter()).chain(rot_slice(rot).iter()) {
                push_f32(out, *value);
            }
        }
        Primitive::Capsule { a, b, r } => {
            push_u32(out, 2);
            for value in a.iter().chain(b.iter()) {
                push_f32(out, *value);
            }
            push_f32(out, *r);
        }
        Primitive::Prism2D { polygon, extent } => {
            push_u32(out, 3);
            push_u32(out, polygon.len() as u32);
            for vertex in polygon {
                push_f32(out, vertex[0]);
                push_f32(out, vertex[1]);
            }
            push_f32(out, extent[0]);
            push_f32(out, extent[1]);
        }
    }
}

fn rot_slice(rot: &Quat) -> [f32; 4] {
    [rot.w, rot.x, rot.y, rot.z]
}

/// A typed view of one JSON object, with the document path for error messages.
struct Reader<'a> {
    path: String,
    entries: &'a [(String, Json)],
}

impl<'a> Reader<'a> {
    fn new(path: impl Into<String>, value: &'a Json) -> Result<Self, SceneError> {
        let path = path.into();
        match value.entries() {
            Some(entries) => Ok(Self { path, entries }),
            None => Err(SceneError::new(path, "expected an object")),
        }
    }

    fn reject_unknown(&self, allowed: &[&str]) -> Result<(), SceneError> {
        for (key, _) in self.entries {
            if !allowed.contains(&key.as_str()) {
                return Err(SceneError::new(
                    self.path.clone(),
                    format!("unknown key `{key}` (allowed here: {})", allowed.join(", ")),
                ));
            }
        }
        Ok(())
    }

    fn path_to(&self, key: &str) -> String {
        format!("{}.{key}", self.path)
    }

    fn optional(&self, key: &str) -> Option<&'a Json> {
        self.entries
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }

    fn get(&self, key: &str) -> Result<&'a Json, SceneError> {
        self.optional(key)
            .ok_or_else(|| SceneError::new(self.path.clone(), format!("missing key `{key}`")))
    }

    fn string(&self, key: &str) -> Result<&'a str, SceneError> {
        self.get(key)?
            .as_str()
            .ok_or_else(|| SceneError::new(self.path_to(key), "expected a string"))
    }

    fn f32(&self, key: &str) -> Result<f32, SceneError> {
        let value = self
            .get(key)?
            .as_f64()
            .ok_or_else(|| SceneError::new(self.path_to(key), "expected a number"))?;
        if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
            return Err(SceneError::new(
                self.path_to(key),
                "number is out of f32 range",
            ));
        }
        Ok(value as f32)
    }

    fn u32(&self, key: &str) -> Result<u32, SceneError> {
        let path = self.path_to(key);
        let value = self
            .get(key)?
            .as_f64()
            .ok_or_else(|| SceneError::new(path.clone(), "expected a number"))?;
        if !value.is_finite() || value < 0.0 || value > f64::from(u32::MAX) || value.fract() != 0.0
        {
            return Err(SceneError::new(path, "expected a non-negative integer"));
        }
        Ok(value as u32)
    }

    fn u64(&self, key: &str) -> Result<u64, SceneError> {
        let path = self.path_to(key);
        let value = self
            .get(key)?
            .as_f64()
            .ok_or_else(|| SceneError::new(path.clone(), "expected a number"))?;
        // 2^53 is the largest integer an f64 holds exactly; anything above it
        // would be a seed this loader cannot reproduce.
        if !value.is_finite()
            || value < 0.0
            || value > 9.007_199_254_740_992e15
            || value.fract() != 0.0
        {
            return Err(SceneError::new(
                path,
                "expected a non-negative integer no larger than 2^53",
            ));
        }
        Ok(value as u64)
    }

    fn array(&self, key: &str) -> Result<&'a [Json], SceneError> {
        self.get(key)?
            .as_array()
            .ok_or_else(|| SceneError::new(self.path_to(key), "expected an array"))
    }

    fn vec2(&self, key: &str) -> Result<[f32; 2], SceneError> {
        let path = self.path_to(key);
        let items = self.array(key)?;
        if items.len() != 2 {
            return Err(SceneError::new(
                path,
                format!("expected 2 values, found {}", items.len()),
            ));
        }
        let mut out = [0.0f32; 2];
        for (axis, item) in items.iter().enumerate() {
            out[axis] = item
                .as_f64()
                .ok_or_else(|| SceneError::new(path.clone(), "expected numbers"))?
                as f32;
        }
        Ok(out)
    }

    fn u32_pair(&self, key: &str) -> Result<[u32; 2], SceneError> {
        let path = self.path_to(key);
        let items = self.array(key)?;
        if items.len() != 2 {
            return Err(SceneError::new(
                path,
                format!("expected 2 values, found {}", items.len()),
            ));
        }
        let mut out = [0u32; 2];
        for (axis, item) in items.iter().enumerate() {
            let value = item
                .as_f64()
                .ok_or_else(|| SceneError::new(path.clone(), "expected numbers"))?;
            if !value.is_finite()
                || value < 0.0
                || value > f64::from(u32::MAX)
                || value.fract() != 0.0
            {
                return Err(SceneError::new(
                    path.clone(),
                    "expected non-negative integers",
                ));
            }
            out[axis] = value as u32;
        }
        Ok(out)
    }

    fn vec3(&self, key: &str) -> Result<[f32; 3], SceneError> {
        let path = self.path_to(key);
        let items = self.array(key)?;
        if items.len() != 3 {
            return Err(SceneError::new(
                path,
                format!("expected 3 values, found {}", items.len()),
            ));
        }
        let mut out = [0.0f32; 3];
        for (axis, item) in items.iter().enumerate() {
            out[axis] = item
                .as_f64()
                .ok_or_else(|| SceneError::new(path.clone(), "expected numbers"))?
                as f32;
        }
        Ok(out)
    }
}

fn parse_solid(value: &Json, path: &str) -> Result<SceneSolid, SceneError> {
    let reader = Reader::new(path, value)?;
    let kind = reader.string("kind")?.to_string();
    reader.reject_unknown(&shape_keys(&kind, true, &reader)?)?;
    let shape = parse_shape_body(&reader, &kind)?;
    Ok(SceneSolid {
        shape,
        rho_g_cm3: reader.f32("rho")?,
    })
}

fn parse_shape(value: &Json, path: &str) -> Result<Primitive, SceneError> {
    let reader = Reader::new(path, value)?;
    let kind = reader.string("kind")?.to_string();
    reader.reject_unknown(&shape_keys(&kind, false, &reader)?)?;
    parse_shape_body(&reader, &kind)
}

fn shape_keys(
    kind: &str,
    with_rho: bool,
    reader: &Reader<'_>,
) -> Result<Vec<&'static str>, SceneError> {
    let keys: &[&'static str] = match kind {
        "box" => {
            if with_rho {
                &["kind", "c", "half", "rot", "rho"]
            } else {
                &["kind", "c", "half", "rot"]
            }
        }
        "ellip" => {
            if with_rho {
                &["kind", "c", "semi", "rot", "rho"]
            } else {
                &["kind", "c", "semi", "rot"]
            }
        }
        "capsule" => {
            if with_rho {
                &["kind", "a", "b", "r", "rho"]
            } else {
                &["kind", "a", "b", "r"]
            }
        }
        "prism2d" => {
            if with_rho {
                &["kind", "polygon", "extent", "rho"]
            } else {
                &["kind", "polygon", "extent"]
            }
        }
        other => {
            return Err(SceneError::new(
                reader.path.clone(),
                format!("unknown primitive kind `{other}` (box, ellip, capsule, prism2d)"),
            ))
        }
    };
    Ok(keys.to_vec())
}

fn parse_shape_body(reader: &Reader<'_>, kind: &str) -> Result<Primitive, SceneError> {
    match kind {
        "box" => Ok(Primitive::Box {
            c: reader.vec3("c")?,
            half: reader.vec3("half")?,
            rot: parse_rot(reader)?,
        }),
        "ellip" => Ok(Primitive::Ellip {
            c: reader.vec3("c")?,
            semi: reader.vec3("semi")?,
            rot: parse_rot(reader)?,
        }),
        "capsule" => Ok(Primitive::Capsule {
            a: reader.vec3("a")?,
            b: reader.vec3("b")?,
            r: reader.f32("r")?,
        }),
        "prism2d" => {
            let polygon = reader
                .array("polygon")?
                .iter()
                .enumerate()
                .map(|(index, vertex)| {
                    let path = format!("{}.polygon[{index}]", reader.path);
                    let items = vertex
                        .as_array()
                        .ok_or_else(|| SceneError::new(path.clone(), "expected an array"))?;
                    if items.len() != 2 {
                        return Err(SceneError::new(path.clone(), "expected 2 values"));
                    }
                    let x = items[0]
                        .as_f64()
                        .ok_or_else(|| SceneError::new(path.clone(), "expected numbers"))?
                        as f32;
                    let y = items[1]
                        .as_f64()
                        .ok_or_else(|| SceneError::new(path.clone(), "expected numbers"))?
                        as f32;
                    Ok([x, y])
                })
                .collect::<Result<Vec<_>, SceneError>>()?;
            Ok(Primitive::Prism2D {
                polygon,
                extent: reader.vec2("extent")?,
            })
        }
        other => Err(SceneError::new(
            reader.path.clone(),
            format!("unknown primitive kind `{other}`"),
        )),
    }
}

/// `rot` is `[x, y, z, w]`, matching the spec's example, and defaults to the
/// identity. Slice mode ignores it; it is carried so 3-D scenes do not have to
/// change shape later.
fn parse_rot(reader: &Reader<'_>) -> Result<Quat, SceneError> {
    let Some(value) = reader.optional("rot") else {
        return Ok(Quat::IDENTITY);
    };
    let path = reader.path_to("rot");
    let items = value
        .as_array()
        .ok_or_else(|| SceneError::new(path.clone(), "expected an array"))?;
    if items.len() != 4 {
        return Err(SceneError::new(
            path.clone(),
            "expected 4 values [x, y, z, w]",
        ));
    }
    let mut parts = [0.0f32; 4];
    for (index, item) in items.iter().enumerate() {
        parts[index] = item
            .as_f64()
            .ok_or_else(|| SceneError::new(path.clone(), "expected numbers"))?
            as f32;
    }
    Ok(Quat::new(parts[3], parts[0], parts[1], parts[2]))
}

fn parse_detector(value: &Json, path: &str) -> Result<Detector, SceneError> {
    let reader = Reader::new(path, value)?;
    reader.reject_unknown(&[
        "pose",
        "area_m2",
        "eff",
        "bins",
        "sigma_ang_mrad",
        "bg_rate_hz",
    ])?;

    let pose = Reader::new(reader.path_to("pose"), reader.get("pose")?)?;
    pose.reject_unknown(&["p", "n", "up"])?;
    let position = pose.vec3("p")?;
    let normal = pose.vec3("n")?;
    let up = match pose.optional("up") {
        Some(value) => {
            let items = value
                .as_array()
                .ok_or_else(|| SceneError::new(pose.path_to("up"), "expected an array"))?;
            if items.len() != 3 {
                return Err(SceneError::new(pose.path_to("up"), "expected 3 values"));
            }
            let mut out = [0.0f32; 3];
            for (axis, item) in items.iter().enumerate() {
                out[axis] = item
                    .as_f64()
                    .ok_or_else(|| SceneError::new(pose.path_to("up"), "expected numbers"))?
                    as f32;
            }
            out
        }
        // Spec §8.1's example gives only `p` and `n`; the roll about the normal
        // is not physics in slice mode, so pick a stable perpendicular.
        None => orthogonal_up(normal),
    };

    Ok(Detector {
        pose: Pose {
            position,
            normal,
            up,
        },
        area_m2: reader.f32("area_m2")?,
        eff: reader.f32("eff")?,
        binning: parse_binning(reader.get("bins")?, &reader.path_to("bins"))?,
        sigma_ang_mrad: reader.f32("sigma_ang_mrad")?,
        bg_rate_hz: reader.f32("bg_rate_hz")?,
    })
}

fn parse_binning(value: &Json, path: &str) -> Result<AngularBinning, SceneError> {
    let reader = Reader::new(path, value)?;
    if reader.optional("flat").is_some() {
        reader.reject_unknown(&["flat"])?;
        let flat = Reader::new(reader.path_to("flat"), reader.get("flat")?)?;
        flat.reject_unknown(&["min", "max", "bins"])?;
        return Ok(AngularBinning::Flat2D {
            min: flat.f32("min")?,
            max: flat.f32("max")?,
            bins: flat.u32("bins")?,
        });
    }
    reader.reject_unknown(&["az", "zen", "az_range", "zen_range"])?;
    let az_range = reader.vec2("az_range")?;
    let zen_range = reader.vec2("zen_range")?;
    Ok(AngularBinning::AzZen {
        az_min: az_range[0],
        az_max: az_range[1],
        az_bins: reader.u32("az")?,
        zen_min: zen_range[0],
        zen_max: zen_range[1],
        zen_bins: reader.u32("zen")?,
    })
}

fn parse_physics(value: &Json) -> Result<ScenePhysics, SceneError> {
    let reader = Reader::new("scene.physics", value)?;
    reader.reject_unknown(&["level", "a", "b", "rho_bulk", "x0"])?;
    let defaults = ScenePhysics::default();
    let level = match reader.string("level")? {
        "L0" => Fidelity::L0,
        "L1" => Fidelity::L1,
        "L2" => Fidelity::L2,
        other => {
            return Err(SceneError::new(
                "scene.physics.level",
                format!("unknown fidelity `{other}` (L0, L1, L2)"),
            ))
        }
    };
    Ok(ScenePhysics {
        level,
        a: reader
            .optional("a")
            .map_or(Ok(defaults.a), |value| f32_of(value, "scene.physics.a"))?,
        b: reader
            .optional("b")
            .map_or(Ok(defaults.b), |value| f32_of(value, "scene.physics.b"))?,
        rho_bulk: reader
            .optional("rho_bulk")
            .map_or(Ok(defaults.rho_bulk), |value| {
                f32_of(value, "scene.physics.rho_bulk")
            })?,
        x0: reader
            .optional("x0")
            .map_or(Ok(defaults.x0), |value| f32_of(value, "scene.physics.x0"))?,
    })
}

fn f32_of(value: &Json, path: &str) -> Result<f32, SceneError> {
    let number = value
        .as_f64()
        .ok_or_else(|| SceneError::new(path, "expected a number"))?;
    if !number.is_finite() || number.abs() > f64::from(f32::MAX) {
        return Err(SceneError::new(path, "number is out of f32 range"));
    }
    Ok(number as f32)
}

/// A unit vector perpendicular to `normal`, or `[1, 0, 0]` for a degenerate
/// normal.
fn orthogonal_up(normal: [f32; 3]) -> [f32; 3] {
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !length.is_finite() || length <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let unit = [normal[0] / length, normal[1] / length, normal[2] / length];
    // Cross with whichever axis is least parallel to the normal.
    let axis = if unit[0].abs() <= unit[1].abs() && unit[0].abs() <= unit[2].abs() {
        [1.0, 0.0, 0.0]
    } else if unit[1].abs() <= unit[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let cross = [
        unit[1] * axis[2] - unit[2] * axis[1],
        unit[2] * axis[0] - unit[0] * axis[2],
        unit[0] * axis[1] - unit[1] * axis[0],
    ];
    let length = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if length > 0.0 {
        [cross[0] / length, cross[1] / length, cross[2] / length]
    } else {
        [1.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed_scene() -> SceneSpec {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../web/src/scenes/default.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        SceneSpec::from_json_str(&text)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn minimal_scene_json(primitives: &str, voids: &str, detectors: &str) -> String {
        format!(
            r#"{{
  "schema": "umbra.scene/1",
  "id": "test",
  "label": "test",
  "units": {{ "length": "m", "density": "g/cm3" }},
  "primitives": {primitives},
  "voids": {voids},
  "detectors": {detectors},
  "exposure_s": 100,
  "grid": {{ "n": [4, 4], "h_m": 1.0, "origin_m": [-2, -2] }},
  "seed": 1
}}"#
        )
    }

    fn detector_json() -> &'static str {
        r#"[{ "pose": { "p": [0.0, -9.0, 0.0], "n": [0, 1, 0] }, "area_m2": 1.0, "eff": 1.0,
             "bins": { "flat": { "min": -0.1, "max": 0.1, "bins": 4 } },
             "sigma_ang_mrad": 0.0, "bg_rate_hz": 0.0 }]"#
    }

    #[test]
    fn the_committed_default_scene_loads_and_validates() {
        let scene = committed_scene();
        assert_eq!(scene.id, "default-slab");
        assert!(
            scene.schematic,
            "the default slab is schematic and must be badged"
        );
        assert_eq!(scene.grid.n, [32, 32]);
        assert_eq!(scene.grid.h_m, 0.5);
        assert_eq!(scene.grid.cell_count(), 1024);
        assert_eq!(scene.primitives.len(), 1);
        assert_eq!(scene.voids.len(), 1);
        assert_eq!(scene.detectors.len(), 1);
        assert_eq!(scene.detectors[0].binning.bins(), 32);
        assert_eq!(scene.physics.level, Fidelity::L1);
        assert_eq!(scene.scene_key_hex().len(), 64);
        assert!(scene.scene_key_hex().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn unknown_keys_are_rejected_and_the_error_names_them() {
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1], "rho": 2.65, "density": 2.65 }]"#,
            "[]",
            detector_json(),
        );
        let error = SceneSpec::from_json_str(&text).expect_err("must reject");
        assert_eq!(error.path, "scene.primitives[0]");
        assert!(error.problem.contains("unknown key `density`"), "{error}");
    }

    #[test]
    fn units_are_checked_rather_than_assumed() {
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1], "rho": 2.65 }]"#,
            "[]",
            detector_json(),
        )
        .replace("\"g/cm3\"", "\"kg/m3\"");
        let error = SceneSpec::from_json_str(&text).expect_err("must reject");
        assert_eq!(error.path, "scene.units.density");
    }

    #[test]
    fn a_missing_or_wrong_schema_is_rejected() {
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1], "rho": 2.65 }]"#,
            "[]",
            detector_json(),
        )
        .replace("umbra.scene/1", "umbra.scene/2");
        let error = SceneSpec::from_json_str(&text).expect_err("must reject");
        assert_eq!(error.path, "scene.schema");
        assert!(error.problem.contains("umbra.scene/1"), "{error}");
    }

    #[test]
    fn geometry_outside_the_grid_is_rejected() {
        // The grid spans [-2, 2] m; a solid centred at x = 10 is invisible to the
        // forward model and must not load quietly.
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [10, 0, 0], "half": [1, 1, 1], "rho": 2.65 }]"#,
            "[]",
            detector_json(),
        );
        let error = SceneSpec::from_json_str(&text).expect_err("must reject");
        assert_eq!(error.path, "scene.primitives[0]");
        assert!(error.problem.contains("outside the grid"), "{error}");
    }

    #[test]
    fn density_adds_solids_and_subtracts_voids() {
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [2, 2, 2], "rho": 2.65 }]"#,
            r#"[{ "kind": "box", "c": [-1.0, -1.0, 0], "half": [0.6, 0.6, 1] }]"#,
            detector_json(),
        );
        let scene = SceneSpec::from_json_str(&text).expect("valid scene");
        let density = scene.density();
        assert_eq!(density.len(), 16);
        // Cell (0, 0) has centre (-1.5, -1.5), inside the void: 2.65 - 2.65 = 0.
        assert_eq!(density[scene.grid.index(0, 0).unwrap()], 0.0);
        // Cell (3, 3) has centre (1.5, 1.5), solid and outside the void.
        assert_eq!(density[scene.grid.index(3, 3).unwrap()], 2.65);
        // Everything is inside the slab, so nothing is empty elsewhere.
        assert!(density.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn a_void_carved_twice_does_not_go_negative() {
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [2, 2, 2], "rho": 2.65 }]"#,
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1] },
                { "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1] }]"#,
            detector_json(),
        );
        let scene = SceneSpec::from_json_str(&text).expect("valid scene");
        let density = scene.density();
        // The two voids cover [-1, 1]^2, so the inner cells are empty and the
        // border cells are still solid. Subtracting twice must not go negative.
        assert_eq!(density[scene.grid.index(1, 1).unwrap()], 0.0);
        assert_eq!(density[scene.grid.index(2, 2).unwrap()], 0.0);
        assert_eq!(density[scene.grid.index(0, 0).unwrap()], 2.65);
        assert!(density.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn the_scene_key_ignores_prose_but_not_physics() {
        let base = committed_scene();
        let mut relabelled = base.clone();
        relabelled.label = "something else entirely".to_string();
        relabelled.notes = Some("edited description".to_string());
        assert_eq!(
            base.scene_key_hex(),
            relabelled.scene_key_hex(),
            "prose is not physics"
        );

        let mut denser = base.clone();
        denser.primitives[0].rho_g_cm3 = 2.66;
        assert_ne!(
            base.scene_key_hex(),
            denser.scene_key_hex(),
            "density is physics"
        );

        let mut longer = base.clone();
        longer.exposure_s *= 2.0;
        assert_ne!(
            base.scene_key_hex(),
            longer.scene_key_hex(),
            "exposure is physics"
        );
    }

    #[test]
    fn replaying_the_same_document_gives_the_same_key_and_field() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../web/src/scenes/default.json"),
        )
        .expect("committed scene is readable");
        let first = SceneSpec::from_json_str(&text).expect("parses");
        let second = SceneSpec::from_json_str(&text).expect("parses");
        assert_eq!(first.scene_key_hex(), second.scene_key_hex());
        assert_eq!(first.scene_key(), second.scene_key());
        assert_eq!(first.density(), second.density());
    }

    #[test]
    fn a_detector_without_an_up_vector_gets_a_perpendicular_one() {
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1], "rho": 2.65 }]"#,
            "[]",
            detector_json(),
        );
        let scene = SceneSpec::from_json_str(&text).expect("valid scene");
        let up = scene.detectors[0].pose.up;
        let normal = scene.detectors[0].pose.normal;
        let dot = up[0] * normal[0] + up[1] * normal[1] + up[2] * normal[2];
        assert!(
            dot.abs() < 1e-6,
            "up must be orthogonal to the normal, got {dot}"
        );
        let length = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
        assert!(
            (length - 1.0).abs() < 1e-6,
            "up must be a unit vector, got {length}"
        );
    }

    #[test]
    fn non_integral_bin_counts_are_rejected() {
        let detectors = detector_json().replace("\"bins\": 4", "\"bins\": 4.5");
        let text = minimal_scene_json(
            r#"[{ "kind": "box", "c": [0, 0, 0], "half": [1, 1, 1], "rho": 2.65 }]"#,
            "[]",
            &detectors,
        );
        let error = SceneSpec::from_json_str(&text).expect_err("must reject");
        assert_eq!(error.path, "scene.detectors[0].bins.flat.bins");
    }

    #[test]
    fn counts_hash_depends_on_every_count_and_its_order() {
        let counts = vec![1u32, 2, 3];
        assert_eq!(counts_hash(&counts).len(), 64);
        assert_ne!(counts_hash(&counts), counts_hash(&[1, 2, 4]));
        assert_ne!(counts_hash(&counts), counts_hash(&[3, 2, 1]));
    }
}
