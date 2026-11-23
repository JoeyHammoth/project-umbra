//! The JavaScript boundary.
//!
//! This crate marshals and nothing else: every algorithm lives in the crates it
//! re-exports, where it can be tested natively (ADR 0002).
//!
//! [`Core`] is the state a worker keeps between tasks: the systems it has
//! assembled and the expected-count rates it has evaluated, both keyed by the
//! **content key the client computed**. The client already hashes the task to
//! look it up in its cache, so letting the façade invent a second key would give
//! the same task two identities.
//!
//! [`EXPECTED`] stores λ as a **rate per second**, not as counts at the scene's
//! exposure. Exposure then stays a free parameter of `SIMULATE` — the slider
//! moves without rebuilding the forward model — and the counts at any exposure
//! are `Poisson(rate · t)`.
//!
//! Bulk data must never cross this boundary through serde: `Vec<f32>` and
//! `Vec<u32>` arrive in JavaScript as `Float32Array` and `Uint32Array`, which
//! the worker then *transfers* to the main thread (spec §4.6).

use std::collections::HashMap;

use umbra_core::{stream_id, Philox};
use umbra_forward::{build_system_2d, expected as expected_counts, SceneSpec, System};
use umbra_physics::{generate_default, Fidelity, FluxTable};
use wasm_bindgen::prelude::*;

/// Version of the compute-core API contract.
///
/// The TypeScript client compares this against its own expectation at startup,
/// so a stale cached `.wasm` fails loudly instead of silently mismatching.
pub const API_VERSION: &str = "0.1.0";

/// Task id mixed into a `SIMULATE` stream id.
///
/// Distinct from the exposure simulator's checkpoint (`1`) and thinning (`2`)
/// ids, so a single-exposure simulation never shares a random stream with a
/// scrubbed one: the two would otherwise draw correlated counts from the same
/// Philox blocks and the difference between them would look like signal.
const TASK_SIMULATE: u32 = 3;

/// Returns the API version this build speaks.
#[wasm_bindgen]
pub fn api_version() -> String {
    API_VERSION.to_string()
}

/// Whether the client's expected version matches this build.
#[wasm_bindgen]
pub fn version_matches(expected: &str) -> bool {
    expected == API_VERSION
}

/// Entry point for the number of workers the pool should use.
///
/// Kept native so the same policy is unit-testable: `hardwareConcurrency − 1`,
/// clamped to `[1, 4]`, which is the worker-pool size in spec §4.4.
pub fn worker_count(hardware_concurrency: u32) -> u32 {
    hardware_concurrency.saturating_sub(1).clamp(1, 4)
}

/// A task the core refused, with a stable code for the UI.
///
/// The code is a fixed identifier (`unknown_system`, `scene`, …), not prose, so
/// the client can branch on it without matching on wording. [`Display`] renders
/// it as `code: message`, which is the shape `web/src/compute/handler.ts`
/// parses back apart.
///
/// [`Display`]: std::fmt::Display
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreError {
    /// Stable identifier for the failure.
    pub code: &'static str,
    /// Human-readable detail, shown to the user.
    pub message: String,
}

impl CoreError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// Result of `BUILD_SYSTEM`, matching `Results.BUILD_SYSTEM` in the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildSummary {
    /// The content key the client supplied.
    pub system_key: String,
    /// Ray cells: one row per detector bin.
    pub rows: u32,
    /// Voxels: the grid size, including voxels no ray touches.
    pub cols: u32,
    /// Non-zero path lengths in the CSR operator.
    pub nnz: u32,
}

struct StoredSystem {
    system: System,
    scene: SceneSpec,
}

struct StoredExpected {
    /// `λ_i / t` in counts per second, so exposure stays a free parameter.
    rate_per_s: Vec<f32>,
    /// The scene the counts came from, mixed into the simulator's stream ids.
    scene_key: u64,
}

/// The stateful compute core behind the façade.
///
/// Everything here is plain Rust: the `wasm-bindgen` layer below it only moves
/// arrays across the boundary, so the behaviour is tested without a browser or a
/// WASM runtime.
#[derive(Default)]
pub struct Core {
    systems: HashMap<String, StoredSystem>,
    expected: HashMap<String, StoredExpected>,
    /// The flux table is ~megabytes of interpolation data; build it once.
    flux: Option<FluxTable>,
}

impl Core {
    /// An empty core: no systems, no expected counts.
    pub fn new() -> Self {
        Self::default()
    }

    /// `BUILD_SYSTEM`: parses the scene document and assembles the path-length
    /// operator.
    ///
    /// The scene crosses the boundary as the §8.1 document, not as a reshaped
    /// JavaScript object, so the strict loader (unknown keys rejected, units
    /// checked, geometry required to lie inside the grid) is the only door in.
    /// Rebuilding a key that is already present is allowed and idempotent: a
    /// content key names one operator, so the second build produces the same
    /// thing.
    pub fn build_system(
        &mut self,
        system_key: &str,
        scene_json: &str,
    ) -> Result<BuildSummary, CoreError> {
        let scene = SceneSpec::from_json_str(scene_json)
            .map_err(|error| CoreError::new("scene", error.to_string()))?;
        let system = build_system_2d(&scene.to_scene_2d());
        let summary = BuildSummary {
            system_key: system_key.to_string(),
            rows: system.ray_count() as u32,
            cols: system.voxel_count() as u32,
            nnz: system.l.indices.len() as u32,
        };
        self.systems
            .insert(system_key.to_string(), StoredSystem { system, scene });
        Ok(summary)
    }

    /// `EXPECTED`: evaluates λ for a density field and remembers the per-second
    /// rate for `SIMULATE`.
    ///
    /// `level` overrides the scene's fidelity for this evaluation only. The
    /// system does not depend on the fidelity — path lengths are geometry — so a
    /// fidelity change invalidates the expected counts and nothing else, which
    /// is why the level travels with this task rather than with `BUILD_SYSTEM`.
    pub fn expected(
        &mut self,
        system_key: &str,
        expected_key: &str,
        density: &[f32],
        level: Option<&str>,
    ) -> Result<Vec<f32>, CoreError> {
        let stored = self.systems.get(system_key).ok_or_else(|| {
            CoreError::new(
                "unknown_system",
                format!("no system for key `{system_key}`: run BUILD_SYSTEM first"),
            )
        })?;
        let fidelity = match level {
            Some(id) => fidelity_from_id(id).ok_or_else(|| {
                CoreError::new("level", format!("unknown fidelity `{id}` (L0, L1, L2)"))
            })?,
            None => stored.scene.physics.level,
        };
        let voxels = stored.system.voxel_count();
        if density.len() < voxels {
            return Err(CoreError::new(
                "density",
                format!(
                    "density has {} values but the grid has {voxels} voxels",
                    density.len()
                ),
            ));
        }
        let exposure = stored.scene.exposure_s;
        let scene_key = stored.scene.scene_key();
        let params = stored.scene.forward_params();
        let flux = self.flux.get_or_insert_with(generate_default);
        let evaluated = expected_counts(&stored.system, density, flux, fidelity, exposure, &params);
        // `exposure > 0` is a scene invariant, so this division cannot blow up.
        let rate_per_s: Vec<f32> = evaluated
            .lambda
            .iter()
            .map(|lambda| lambda / exposure)
            .collect();
        self.expected.insert(
            expected_key.to_string(),
            StoredExpected {
                rate_per_s,
                scene_key,
            },
        );
        Ok(evaluated.lambda)
    }

    /// `SIMULATE`: Poisson counts at an arbitrary exposure.
    ///
    /// Counts are drawn per ray from a stream keyed by `(seed, scene, ray)`, so
    /// the same seed reproduces the same dataset however the tasks were
    /// scheduled — the counter-based contract of spec §4.7 — while a different
    /// scene with the same seed is an independent universe.
    pub fn simulate(
        &mut self,
        expected_key: &str,
        seed: u64,
        exposure_s: f32,
    ) -> Result<Vec<u32>, CoreError> {
        let stored = self.expected.get(expected_key).ok_or_else(|| {
            CoreError::new(
                "unknown_expected",
                format!("no expected counts for key `{expected_key}`: run EXPECTED first"),
            )
        })?;
        if !exposure_s.is_finite() || exposure_s < 0.0 {
            return Err(CoreError::new(
                "exposure",
                format!("exposure {exposure_s} s must be finite and non-negative"),
            ));
        }
        if exposure_s == 0.0 {
            return Ok(vec![0; stored.rate_per_s.len()]);
        }
        let mut counts = Vec::with_capacity(stored.rate_per_s.len());
        for (ray, rate) in stored.rate_per_s.iter().enumerate() {
            let mean = f64::from(*rate) * f64::from(exposure_s);
            let stream = stream_id(stored.scene_key, TASK_SIMULATE, ray as u64);
            let mut rng = Philox::new(seed, stream);
            counts.push(rng.poisson(mean.max(0.0)).min(u64::from(u32::MAX)) as u32);
        }
        Ok(counts)
    }

    /// Number of systems the core is holding, for the worker's own accounting.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }
}

fn fidelity_from_id(id: &str) -> Option<Fidelity> {
    match id {
        "L0" => Some(Fidelity::L0),
        "L1" => Some(Fidelity::L1),
        "L2" => Some(Fidelity::L2),
        _ => None,
    }
}

/// The façade a worker holds for its lifetime.
#[wasm_bindgen]
pub struct ComputeCore {
    core: Core,
}

impl Default for ComputeCore {
    fn default() -> Self {
        Self { core: Core::new() }
    }
}

/// `BUILD_SYSTEM` returned across the boundary.
#[wasm_bindgen]
pub struct BuildOutcome {
    system_key: String,
    rows: u32,
    cols: u32,
    nnz: u32,
}

#[wasm_bindgen]
impl BuildOutcome {
    /// The content key the client supplied.
    #[wasm_bindgen(getter, js_name = systemKey)]
    pub fn system_key(&self) -> String {
        self.system_key.clone()
    }

    /// Ray cells.
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Voxels.
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Non-zero path lengths.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> u32 {
        self.nnz
    }
}

impl From<BuildSummary> for BuildOutcome {
    fn from(summary: BuildSummary) -> Self {
        Self {
            system_key: summary.system_key,
            rows: summary.rows,
            cols: summary.cols,
            nnz: summary.nnz,
        }
    }
}

fn fault(error: CoreError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
impl ComputeCore {
    /// A core with nothing built yet.
    #[wasm_bindgen(constructor)]
    pub fn new() -> ComputeCore {
        ComputeCore::default()
    }

    /// `BUILD_SYSTEM`: `scene_json` is the §8.1 scene document.
    ///
    /// Returns the operator's shape; the operator itself stays in the worker,
    /// where the next task can use it without crossing the boundary again.
    #[wasm_bindgen(js_name = buildSystem)]
    pub fn build_system(
        &mut self,
        system_key: &str,
        scene_json: &str,
    ) -> Result<BuildOutcome, JsValue> {
        self.core
            .build_system(system_key, scene_json)
            .map(BuildOutcome::from)
            .map_err(fault)
    }

    /// `EXPECTED`: `density` is a `Float32Array` of g cm⁻³; the result is λ as a
    /// `Float32Array`, transferred by the worker rather than serialised.
    #[wasm_bindgen(js_name = expected)]
    pub fn expected(
        &mut self,
        system_key: &str,
        expected_key: &str,
        density: &[f32],
        level: Option<String>,
    ) -> Result<Vec<f32>, JsValue> {
        self.core
            .expected(system_key, expected_key, density, level.as_deref())
            .map_err(fault)
    }

    /// `SIMULATE`: counts at `exposure_s`, as a `Uint32Array`.
    #[wasm_bindgen(js_name = simulate)]
    pub fn simulate(
        &mut self,
        expected_key: &str,
        seed: u64,
        exposure_s: f32,
    ) -> Result<Vec<u32>, JsValue> {
        self.core
            .simulate(expected_key, seed, exposure_s)
            .map_err(fault)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::Grid2;

    /// A small slab in a 16 × 16 grid of 0.5 m voxels, with `detectors`
    /// fan-shaped detectors of `bins` rays each.
    fn scene_json(exposure_s: f32, bins: u32, detectors: usize) -> String {
        let mut heads = Vec::new();
        for detector in 0..detectors {
            let x = -3.0 + detector as f32;
            heads.push(format!(
                r#"{{ "pose": {{ "p": [{x}, -9.0, 0.0], "n": [0, 1, 0] }},
                     "area_m2": 1.0, "eff": 1.0,
                     "bins": {{ "flat": {{ "min": -0.4, "max": 0.4, "bins": {bins} }} }},
                     "sigma_ang_mrad": 0.0, "bg_rate_hz": 0.0 }}"#
            ));
        }
        format!(
            r#"{{
  "schema": "umbra.scene/1",
  "id": "wasm-test",
  "label": "wasm test",
  "schematic": true,
  "units": {{ "length": "m", "density": "g/cm3" }},
  "primitives": [ {{ "kind": "box", "c": [0, 0, 0], "half": [3, 2, 2], "rho": 2.65 }} ],
  "detectors": [{}],
  "exposure_s": {exposure_s},
  "grid": {{ "n": [16, 16], "h_m": 0.5, "origin_m": [-4, -4] }},
  "seed": 7
}}"#,
            heads.join(",")
        )
    }

    fn scene(exposure_s: f32, bins: u32, detectors: usize) -> SceneSpec {
        let text = scene_json(exposure_s, bins, detectors);
        SceneSpec::from_json_str(&text).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn version_handshake_round_trips() {
        assert_eq!(api_version(), API_VERSION);
        assert!(version_matches(API_VERSION));
        assert!(!version_matches("0.0.1"));
    }

    #[test]
    fn worker_pool_is_clamped() {
        assert_eq!(worker_count(1), 1);
        assert_eq!(worker_count(8), 4);
        assert_eq!(worker_count(4), 3);
    }

    #[test]
    fn build_system_reports_the_shape_of_the_operator_it_built() {
        let text = scene_json(100.0, 8, 3);
        let direct = build_system_2d(&scene(100.0, 8, 3).to_scene_2d());
        let mut core = Core::new();
        let summary = core.build_system("sys", &text).expect("build");
        assert_eq!(summary.system_key, "sys");
        assert_eq!(summary.rows as usize, direct.ray_count());
        assert_eq!(summary.cols as usize, direct.voxel_count());
        assert_eq!(summary.nnz as usize, direct.l.indices.len());
        assert_eq!(summary.rows, 24, "three detectors of eight bins");
        assert_eq!(summary.cols, 256);
        assert_eq!(core.system_count(), 1);

        // Building the same key again is idempotent: a content key names one
        // operator.
        let again = core.build_system("sys", &text).expect("rebuild");
        assert_eq!(again, summary);
        assert_eq!(core.system_count(), 1);
    }

    #[test]
    fn expected_matches_a_direct_forward_evaluation() {
        let text = scene_json(1000.0, 8, 2);
        let parsed = scene(1000.0, 8, 2);
        let density = parsed.density();
        let system = build_system_2d(&parsed.to_scene_2d());
        let direct = expected_counts(
            &system,
            &density,
            &generate_default(),
            parsed.physics.level,
            parsed.exposure_s,
            &parsed.forward_params(),
        );

        let mut core = Core::new();
        core.build_system("sys", &text).expect("build");
        let lambda = core
            .expected("sys", "exp", &density, None)
            .expect("expected");
        assert_eq!(
            lambda, direct.lambda,
            "the façade must not alter the physics"
        );
        assert_eq!(lambda.len(), 16);
        assert!(lambda.iter().all(|value| value.is_finite() && *value > 0.0));
    }

    #[test]
    fn a_fidelity_override_changes_the_counts_and_not_the_system() {
        let text = scene_json(1000.0, 4, 1);
        let density = scene(1000.0, 4, 1).density();
        let mut core = Core::new();
        core.build_system("sys", &text).expect("build");
        let l0 = core
            .expected("sys", "exp-l0", &density, Some("L0"))
            .expect("L0");
        let l1 = core
            .expected("sys", "exp-l1", &density, Some("L1"))
            .expect("L1");
        assert_ne!(l0, l1, "the level must select a different model");
        assert_eq!(core.system_count(), 1, "the operator is level-independent");
    }

    #[test]
    fn counts_carry_the_poisson_scatter_their_mean_claims() {
        // 24 rays at ~10^3 counts each: the total should sit within a few sigma
        // of Σλ. A simulator that returned λ itself, or one with the wrong
        // exposure scaling, fails this immediately.
        let text = scene_json(3600.0, 8, 3);
        let density = scene(3600.0, 8, 3).density();
        let mut core = Core::new();
        core.build_system("sys", &text).expect("build");
        let lambda = core
            .expected("sys", "exp", &density, None)
            .expect("expected");
        let expected_total: f64 = lambda.iter().map(|value| f64::from(*value)).sum();
        assert!(
            expected_total > 100.0,
            "the test scene must produce a measurable number of counts, got {expected_total}"
        );

        let counts = core.simulate("exp", 1234, 3600.0).expect("simulate");
        assert_eq!(counts.len(), lambda.len());
        let total: f64 = counts.iter().map(|count| f64::from(*count)).sum();
        let sigma = expected_total.sqrt();
        assert!(
            (total - expected_total).abs() < 4.0 * sigma,
            "total {total} vs expected {expected_total} (sigma {sigma})"
        );
        let distinct: std::collections::HashSet<u32> = counts.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "counts are identical: no scatter at all"
        );
    }

    #[test]
    fn simulation_is_reproducible_for_a_seed_and_independent_between_seeds() {
        let text = scene_json(3600.0, 8, 3);
        let density = scene(3600.0, 8, 3).density();
        let mut core = Core::new();
        core.build_system("sys", &text).expect("build");
        core.expected("sys", "exp", &density, None)
            .expect("expected");

        let first = core.simulate("exp", 99, 3600.0).expect("simulate");
        let again = core.simulate("exp", 99, 3600.0).expect("simulate");
        assert_eq!(first, again, "the same seed must replay the same dataset");
        let other = core.simulate("exp", 100, 3600.0).expect("simulate");
        assert_ne!(first, other, "a different seed must be a different dataset");
    }

    #[test]
    fn zero_exposure_is_zero_counts_and_the_rate_scales_with_exposure() {
        let text = scene_json(3600.0, 8, 2);
        let density = scene(3600.0, 8, 2).density();
        let mut core = Core::new();
        core.build_system("sys", &text).expect("build");
        let lambda = core
            .expected("sys", "exp", &density, None)
            .expect("expected");

        assert_eq!(
            core.simulate("exp", 5, 0.0).expect("simulate"),
            vec![0; lambda.len()]
        );
        // Half the exposure halves the mean, because the rate is stored per
        // second rather than as counts at the scene's exposure.
        let half = core.simulate("exp", 5, 1800.0).expect("simulate");
        let full_total: f64 = lambda.iter().map(|value| f64::from(*value)).sum();
        let half_total: f64 = half.iter().map(|count| f64::from(*count)).sum();
        let sigma = (full_total * 0.5).sqrt();
        assert!(
            (half_total - full_total * 0.5).abs() < 4.0 * sigma,
            "half-exposure total {half_total} vs {}",
            full_total * 0.5
        );
    }

    #[test]
    fn a_bad_request_is_a_coded_error_rather_than_a_panic() {
        let mut core = Core::new();
        let missing = core
            .expected("nope", "exp", &[1.0], None)
            .expect_err("unknown system");
        assert_eq!(missing.code, "unknown_system");

        let unknown_expected = core
            .simulate("nope", 1, 10.0)
            .expect_err("unknown expected");
        assert_eq!(unknown_expected.code, "unknown_expected");

        let text = scene_json(100.0, 2, 1);
        core.build_system("sys", &text).expect("build");
        let short = core
            .expected("sys", "exp", &[2.65, 2.65], None)
            .expect_err("short density");
        assert_eq!(short.code, "density");
        assert!(
            short.message.contains("256 voxels"),
            "message must name the grid: {}",
            short.message
        );

        let level = core
            .expected("sys", "exp", &vec![2.65; 256], Some("L9"))
            .expect_err("unknown fidelity");
        assert_eq!(level.code, "level");

        let negative = core
            .expected("sys", "exp", &vec![2.65; 256], None)
            .and_then(|_| core.simulate("exp", 1, -1.0))
            .expect_err("negative exposure");
        assert_eq!(negative.code, "exposure");
    }

    #[test]
    fn a_scene_the_loader_rejects_comes_back_as_a_scene_error() {
        let mut core = Core::new();
        // Density in kg/m3 is the classic silent factor of a thousand.
        let text = scene_json(100.0, 2, 1).replace("g/cm3", "kg/m3");
        let error = core.build_system("sys", &text).expect_err("bad units");
        assert_eq!(error.code, "scene");
        assert!(
            error.message.contains("units.density"),
            "the loader's field path must survive: {}",
            error.message
        );
        assert_eq!(core.system_count(), 0, "nothing may be stored");
    }

    #[test]
    fn the_error_display_is_the_code_then_the_message() {
        let error = CoreError::new("density", "3 values but the grid has 256 voxels");
        assert_eq!(
            error.to_string(),
            "density: 3 values but the grid has 256 voxels"
        );
        // The grid in this module's scenes is 16 × 16 of 0.5 m voxels.
        let grid = Grid2::new([16, 16], [-4.0, -4.0], 0.5);
        assert_eq!(grid.cell_count(), 256);
    }
}
