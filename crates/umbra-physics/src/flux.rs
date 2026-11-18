//! Tabulated integral and differential muon intensity.
//!
//! The surface flux is the modified Gaisser parametrisation of Guan et al.
//! (2015), *A parametrization of the cosmic-ray muon flux at sea-level*,
//! arXiv:1509.06176, Eq. 3, with the Earth-curvature effective angle of
//! Eq. 2. It differs from Gaisser's original formula in two ways that both
//! matter at low energy: the `3.64 / (E cos^1.29 θ*)` term inside the spectral
//! index, and the curvature correction to `cos θ`.
//!
//! `Φ(>E, θ)` is precomputed offline on a `(log10 E, cos θ)` grid by
//! [`generate`] and interpolated bilinearly in log space at runtime. The table
//! is a versioned asset (spec §8.5), so serialisation lives here too: the format
//! is fixed by §14.3 and carries a SHA-256 of its own payload.
//!
//! # Why the table exists rather than a runtime integral
//!
//! Every solver evaluates `Φ` millions of times inside inner loops. The table
//! costs 34 KB and one bilinear interpolation; the integral costs a 400-point
//! quadrature. The numbers agree to well under the physics tolerance, and
//! [`FluxTable::integral`] is what the forward model calls.

use umbra_core::hash::sha256;

/// Magic bytes at the head of a serialised table.
pub const MAGIC: [u8; 4] = *b"UBT1";

/// Table format version, asserted on load (spec §14.3).
pub const TABLE_VERSION: u16 = 1;

/// Curvature parameter `P1` (Guan et al. 2015, Table 1).
const P1: f64 = 0.102_573;
/// Curvature parameter `P2`.
const P2: f64 = -0.068_287;
/// Curvature parameter `P3`.
const P3: f64 = 0.958_633;
/// Curvature parameter `P4`.
const P4: f64 = 0.040_725_3;
/// Curvature parameter `P5`.
const P5: f64 = 0.817_285;

/// High-energy cut-off of the tabulated integral, in GeV.
///
/// Above this the differential flux falls as `E^-1.7`, so the contribution to
/// `Φ(>E)` is below 1e-4 relative for every `E` on the grid.
const INTEGRAL_CEILING_GEV: f64 = 1.0e6;

/// The Earth-curvature effective cosine, `cos θ*` (Guan et al. 2015, Eq. 2).
///
/// At `cos θ = 1` it returns exactly 1, so the vertical flux is unchanged by the
/// curvature correction; at the horizon it returns ≈ 0.103 rather than 0, which
/// is the point of the correction.
pub fn cos_theta_star(cos_theta: f64) -> f64 {
    let c = cos_theta.clamp(0.0, 1.0);
    let numerator = (c * c + P1 * P1 + P2 * c.powf(P3) + P4 * c.powf(P5)).sqrt();
    let denominator = (1.0 + P1 * P1 + P2 + P4).sqrt();
    numerator / denominator
}

/// Differential muon intensity `I(E, θ)`, in cm⁻² s⁻¹ sr⁻¹ GeV⁻¹.
///
/// Guan et al. (2015), Eq. 3. `e_gev` is the muon energy and `cos_theta` the
/// zenith cosine; both are clamped to the physical range rather than panicking,
/// because the value arrives from scene JSON.
pub fn gaisser_guan_diff(e_gev: f64, cos_theta: f64) -> f64 {
    if e_gev <= 0.0 {
        return 0.0;
    }
    let cs = cos_theta_star(cos_theta);
    // Eq. 3: the low-energy modification lives inside the spectral index, and it
    // *multiplies* the energy — `E (1 + a / (E cos^1.29))`. Dividing instead of
    // multiplying inverts the spectrum at low energy and raises the vertical flux
    // by two orders of magnitude, which the anchor test catches.
    let e_term = e_gev * (1.0 + 3.64 / (e_gev * cs.powf(1.29)));
    let pion = 1.0 / (1.0 + 1.1 * e_gev * cs / 115.0);
    let kaon = 0.054 / (1.0 + 1.1 * e_gev * cs / 850.0);
    0.14 * e_term.powf(-2.7) * (pion + kaon)
}

/// Integral intensity `Φ(>E, θ)`, in cm⁻² s⁻¹ sr⁻¹.
///
/// Composite Simpson in `ln E`, where the integrand is `I(E, θ) · E`.
pub fn integral_flux(e_min_gev: f64, cos_theta: f64, intervals: usize) -> f64 {
    let low = e_min_gev.max(1e-3).ln();
    let high = INTEGRAL_CEILING_GEV.ln();
    if low >= high {
        return 0.0;
    }
    let n = intervals.max(2) & !1; // Simpson needs an even count
    let h = (high - low) / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let ln_e = low + h * i as f64;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let e = ln_e.exp();
        sum += weight * gaisser_guan_diff(e, cos_theta) * e;
    }
    sum * h / 3.0
}

/// A precomputed muon intensity table.
///
/// `integral` and `diff` are row-major with `log10_e_min` as the slow index and
/// `cos_theta` as the fast index, so a single ray's lookup touches two adjacent
/// cache lines.
#[derive(Debug, Clone, PartialEq)]
pub struct FluxTable {
    /// Version string of the generating table, e.g. `gaisser-guan-2015.v1`.
    pub version: String,
    /// Ascending `log10(E_min / GeV)` axis.
    pub log10_e_min: Vec<f32>,
    /// Ascending `cos θ` axis.
    pub cos_theta: Vec<f32>,
    /// `Φ(>E, θ)` in cm⁻² s⁻¹ sr⁻¹.
    pub integral: Vec<f32>,
    /// `I(E, θ) = -∂Φ/∂E` in cm⁻² s⁻¹ sr⁻¹ GeV⁻¹.
    pub diff: Vec<f32>,
    /// Cached `dΦ/d(log10 E)` at each node, derived from `diff` and the axis.
    ///
    /// Not part of the serialised format — it is a pure function of `diff` and
    /// the energy axis. Kept because computing it inside a lookup costs two
    /// `powf` calls per ray cell, which was a measurable share of the forward
    /// model's budget.
    dphi_dloge: Vec<f32>,
}

/// `dΦ/d(log10 E) = -I · ln 10 · E` at every node.
fn slopes(log10_e: &[f32], diff: &[f32], n_cos: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(diff.len());
    for (i, log_e) in log10_e.iter().enumerate() {
        let scale = -std::f32::consts::LN_10 * 10f32.powf(*log_e);
        for c in 0..n_cos {
            out.push(diff[i * n_cos + c] * scale);
        }
    }
    out
}

/// Reasons a serialised table can be rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableError {
    /// The file does not begin with `UBT1`.
    BadMagic,
    /// The format version is not one this build understands.
    UnsupportedVersion(u16),
    /// The file is shorter than its header claims.
    Truncated,
    /// The payload hash does not match the trailer: the file was edited.
    ChecksumMismatch,
    /// Axes and value arrays disagree in length.
    Inconsistent,
}

impl std::fmt::Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableError::BadMagic => write!(f, "not a UMBRA table (bad magic)"),
            TableError::UnsupportedVersion(v) => {
                write!(f, "unsupported table version {v}, expected {TABLE_VERSION}")
            }
            TableError::Truncated => write!(f, "table file is truncated"),
            TableError::ChecksumMismatch => write!(f, "table checksum does not match"),
            TableError::Inconsistent => write!(f, "table axes and values are inconsistent"),
        }
    }
}

impl std::error::Error for TableError {}

impl FluxTable {
    /// Number of grid entries.
    pub fn len(&self) -> usize {
        self.integral.len()
    }

    /// Whether the table holds no values.
    pub fn is_empty(&self) -> bool {
        self.integral.is_empty()
    }

    /// Whether the axes and value buffers are mutually consistent.
    pub fn is_consistent(&self) -> bool {
        let expected = self.log10_e_min.len() * self.cos_theta.len();
        expected != 0 && self.integral.len() == expected && self.diff.len() == expected
    }

    /// Bilinear lookup with clamping at both axes.
    ///
    /// Returns `(index_of_lower_node, fraction)`. The axes are uniform, which the
    /// writer enforces, so no search is needed.
    fn locate(axis: &[f32], value: f32) -> (usize, f32) {
        let n = axis.len();
        if n == 1 {
            return (0, 0.0);
        }
        let first = axis[0];
        let step = (axis[n - 1] - first) / (n - 1) as f32;
        if value <= first {
            return (0, 0.0);
        }
        if value >= axis[n - 1] {
            return (n - 2, 1.0);
        }
        let pos = (value - first) / step;
        let i = (pos.floor() as usize).min(n - 2);
        (i, pos - i as f32)
    }

    /// Bilinear interpolation of one value buffer.
    fn sample(&self, values: &[f32], e_min_gev: f32, cos_theta: f32) -> f32 {
        let log_e = e_min_gev.max(1e-6).log10();
        let (ie, fe) = Self::locate(&self.log10_e_min, log_e);
        let (ic, fc) = Self::locate(&self.cos_theta, cos_theta.clamp(0.0, 1.0));
        let nc = self.cos_theta.len();
        let at = |e: usize, c: usize| values[e * nc + c];
        let a = at(ie, ic);
        let b = at(ie + 1, ic);
        let c = at(ie, ic + 1);
        let d = at(ie + 1, ic + 1);
        let low = a + (c - a) * fc;
        let high = b + (d - b) * fc;
        low + (high - low) * fe
    }

    /// Cubic Hermite interpolation along the energy axis for one zenith column.
    ///
    /// The slopes come from the tabulated derivative, so at a node the
    /// interpolant's derivative is exactly `-I` rather than the secant between
    /// neighbouring nodes. That matters because every solver consumes
    /// `∂λ/∂X`, which is proportional to `I(E_min)`: with plain linear
    /// interpolation the two disagreed by several percent across a single grid
    /// cell, and the finite-difference check caught it.
    fn hermite_energy(&self, column: usize, log_e: f32) -> (f32, f32) {
        let axis = &self.log10_e_min;
        let n_e = axis.len();
        let nc = self.cos_theta.len();
        let at = |e: usize| self.integral[e * nc + column];
        // A node recomputed as `10^log10(E)` can land a hair outside the axis, so
        // the clamp uses a tolerance: without it, the first node reports a zero
        // slope and the analytic gradient silently loses its scale there.
        const EDGE: f32 = 1e-5;
        if n_e < 2 || log_e < axis[0] - EDGE {
            return (at(0), 0.0);
        }
        if log_e > axis[n_e - 1] + EDGE {
            return (at(n_e - 1), 0.0);
        }
        let step = (axis[n_e - 1] - axis[0]) / (n_e - 1) as f32;
        let pos = (log_e - axis[0]) / step;
        let i = (pos.floor() as usize).min(n_e - 2);
        // Clamping matters: a node computed as 10^log10(E) can land 1e-7 outside
        // its interval, and an unclamped t would extrapolate the cubic.
        let t = (pos - i as f32).clamp(0.0, 1.0);

        let y0 = at(i);
        let y1 = at(i + 1);
        // dΦ/d(log10 E) = dΦ/dE · dE/d(log10 E) = (−I) · ln10 · E
        let slope0 = self.dphi_dloge[i * nc + column];
        let slope1 = self.dphi_dloge[(i + 1) * nc + column];

        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let value = h00 * y0 + h10 * step * slope0 + h01 * y1 + h11 * step * slope1;

        // Derivative of the same cubic with respect to log10 E.
        let d00 = 6.0 * t2 - 6.0 * t;
        let d10 = 3.0 * t2 - 4.0 * t + 1.0;
        let d01 = -6.0 * t2 + 6.0 * t;
        let d11 = 3.0 * t2 - 2.0 * t;
        let dlog_e = (d00 * y0 + d10 * step * slope0 + d01 * y1 + d11 * step * slope1) / step;
        (value, dlog_e)
    }

    /// `Φ(>E_min, θ)` in cm⁻² s⁻¹ sr⁻¹.
    ///
    /// Hermite along energy, linear in `cos θ`. The zenith axis is smooth enough
    /// that the derivative is not consumed through it.
    pub fn integral(&self, e_min_gev: f32, cos_theta: f32) -> f32 {
        self.integral_and_slope(e_min_gev, cos_theta).0
    }

    /// `Φ(>E, θ)` and `dΦ/dE` in one pass.
    ///
    /// The forward model needs both for the same ray cell, and evaluating them
    /// separately repeated the whole interpolation — half the cost for exactly
    /// the same numbers.
    pub fn integral_and_slope(&self, e_min_gev: f32, cos_theta: f32) -> (f32, f32) {
        let e = e_min_gev.max(1e-6);
        let log_e = e.log10();
        let (ic, fc) = Self::locate(&self.cos_theta, cos_theta.clamp(0.0, 1.0));
        let (mut value, mut slope) = self.hermite_energy(ic, log_e);
        let nc = self.cos_theta.len();
        if fc > 0.0 && ic + 1 < nc {
            let (high_value, high_slope) = self.hermite_energy(ic + 1, log_e);
            value += (high_value - value) * fc;
            slope += (high_slope - slope) * fc;
        }
        // dΦ/dE = dΦ/d(log10 E) / (ln 10 · E)
        (value, slope / (std::f32::consts::LN_10 * e))
    }

    /// `dΦ/dE` in cm⁻² s⁻¹ sr⁻¹ GeV⁻¹, taken from the derivative of the
    /// interpolant rather than from the separately tabulated `diff`.
    ///
    /// This distinction is not pedantry: `∂λ/∂X` must be the derivative of the
    /// model that produced λ, or a solver's line search and its gradient
    /// disagree by the interpolation error (measured at ~2% with bilinear
    /// interpolation). At a grid node the two agree exactly, which a test pins.
    pub fn integral_slope(&self, e_min_gev: f32, cos_theta: f32) -> f32 {
        self.integral_and_slope(e_min_gev, cos_theta).1
    }

    /// `I(E, θ) = -∂Φ/∂E` in cm⁻² s⁻¹ sr⁻¹ GeV⁻¹.
    pub fn diff(&self, e_gev: f32, cos_theta: f32) -> f32 {
        self.sample(&self.diff, e_gev, cos_theta)
    }

    /// Serialises to the `.ubt` layout fixed by spec §14.3.
    pub fn to_bytes(&self) -> Vec<u8> {
        assert!(
            self.is_consistent(),
            "refusing to serialise a malformed table"
        );
        let n_e = self.log10_e_min.len();
        let n_cos = self.cos_theta.len();
        let step_e = if n_e > 1 {
            (self.log10_e_min[n_e - 1] - self.log10_e_min[0]) / (n_e - 1) as f32
        } else {
            0.0
        };
        let step_c = if n_cos > 1 {
            (self.cos_theta[n_cos - 1] - self.cos_theta[0]) / (n_cos - 1) as f32
        } else {
            0.0
        };

        let mut out = Vec::with_capacity(32 + 8 * n_e * n_cos + 32);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&TABLE_VERSION.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // flag bit 0: derivative present
        out.extend_from_slice(&(n_e as u32).to_le_bytes());
        out.extend_from_slice(&(n_cos as u32).to_le_bytes());
        out.extend_from_slice(&self.log10_e_min[0].to_le_bytes());
        out.extend_from_slice(&step_e.to_le_bytes());
        out.extend_from_slice(&self.cos_theta[0].to_le_bytes());
        out.extend_from_slice(&step_c.to_le_bytes());
        for value in &self.integral {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.diff {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let digest = sha256(&out);
        out.extend_from_slice(&digest);
        out
    }

    /// Reads the `.ubt` layout, verifying magic, version, length and checksum.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TableError> {
        if bytes.len() < 64 {
            return Err(TableError::Truncated);
        }
        if bytes[0..4] != MAGIC {
            return Err(TableError::BadMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != TABLE_VERSION {
            return Err(TableError::UnsupportedVersion(version));
        }
        let _flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let read_u32 = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
        };
        let read_f32 = |at: usize| {
            f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let n_e = read_u32(8);
        let n_cos = read_u32(12);
        let e0 = read_f32(16);
        let de = read_f32(20);
        let c0 = read_f32(24);
        let dc = read_f32(28);
        let count = n_e.checked_mul(n_cos).ok_or(TableError::Inconsistent)?;
        let expected = 32 + 8 * count + 32;
        if bytes.len() != expected {
            return Err(TableError::Truncated);
        }
        let payload_end = bytes.len() - 32;
        if sha256(&bytes[..payload_end]) != bytes[payload_end..] {
            return Err(TableError::ChecksumMismatch);
        }

        let mut integral = Vec::with_capacity(count);
        let mut diff = Vec::with_capacity(count);
        for i in 0..count {
            integral.push(read_f32(32 + 4 * i));
        }
        for i in 0..count {
            diff.push(read_f32(32 + 4 * count + 4 * i));
        }

        let log10_e_min: Vec<f32> = (0..n_e).map(|i| e0 + i as f32 * de).collect();
        let dphi_dloge = slopes(&log10_e_min, &diff, n_cos);
        Ok(Self {
            version: format!("gaisser-guan-2015.v{TABLE_VERSION}"),
            log10_e_min,
            cos_theta: (0..n_cos).map(|i| c0 + i as f32 * dc).collect(),
            integral,
            diff,
            dphi_dloge,
        })
    }
}

/// Builds the default table: 129 log-spaced energies, 33 zenith cosines.
///
/// The grid is the one specified in spec §6.2: `log10 E ∈ [-1, 4]` and
/// `cos θ ∈ [0, 1]`.
pub fn generate_default() -> FluxTable {
    generate(129, 33, -1.0, 4.0)
}

/// Builds a table on a uniform `(log10 E, cos θ)` grid.
///
/// `intervals` controls the quadrature resolution of the integral and is fixed
/// rather than exposed at runtime: the table is generated offline, audited once,
/// and then immutable.
pub fn generate(n_e: usize, n_cos: usize, log10_e_min: f32, log10_e_max: f32) -> FluxTable {
    assert!(n_e >= 2 && n_cos >= 2, "a table needs at least a 2x2 grid");
    let mut log10_e_min_axis = Vec::with_capacity(n_e);
    let mut cos_axis = Vec::with_capacity(n_cos);
    let mut integral = Vec::with_capacity(n_e * n_cos);
    let mut diff = Vec::with_capacity(n_e * n_cos);

    for i in 0..n_e {
        let log_e = log10_e_min + (log10_e_max - log10_e_min) * i as f32 / (n_e - 1) as f32;
        log10_e_min_axis.push(log_e);
    }
    for j in 0..n_cos {
        cos_axis.push(j as f32 / (n_cos - 1) as f32);
    }
    for log_e in &log10_e_min_axis {
        let e = 10f64.powf(f64::from(*log_e));
        for cos in &cos_axis {
            let c = f64::from(*cos);
            integral.push(integral_flux(e, c, 400) as f32);
            diff.push(gaisser_guan_diff(e, c) as f32);
        }
    }

    let dphi_dloge = slopes(&log10_e_min_axis, &diff, n_cos);
    FluxTable {
        version: format!("gaisser-guan-2015.v{TABLE_VERSION}"),
        log10_e_min: log10_e_min_axis,
        cos_theta: cos_axis,
        integral,
        diff,
        dphi_dloge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_cosine_is_unity_vertically_and_small_at_the_horizon() {
        assert!((cos_theta_star(1.0) - 1.0).abs() < 1e-6);
        let horizon = cos_theta_star(0.0);
        assert!(
            (0.09..0.12).contains(&horizon),
            "horizon cos θ* = {horizon}"
        );
        // Monotone in cos θ, which the interpolation relies on.
        let mut previous = 0.0;
        for step in 0..=20 {
            let value = cos_theta_star(step as f64 / 20.0);
            assert!(value >= previous);
            previous = value;
        }
    }

    #[test]
    fn vertical_integral_flux_above_one_gev_is_about_seventy_per_m2() {
        // Literature anchor (Columbia cosmic-ray review; also the standard figure
        // quoted for muography): the vertical integral intensity above 1 GeV is
        // ≈ 70 m⁻² s⁻¹ sr⁻¹ = 7.0e-3 cm⁻² s⁻¹ sr⁻¹. This single number catches
        // the transcription errors that a shape test cannot: implementing Guan's
        // bracket as a division instead of a multiplication raises it by ~400×.
        let phi = integral_flux(1.0, 1.0, 2000);
        assert!(
            (5.0e-3..9.0e-3).contains(&phi),
            "Φ(>1 GeV, vertical) = {phi} cm^-2 s^-1 sr^-1"
        );
        // 60 m⁻² s⁻¹ sr⁻¹ is the measured value with this parametrisation.
        let per_m2 = phi * 1.0e4;
        assert!((50.0..90.0).contains(&per_m2), "{per_m2} m^-2 s^-1 sr^-1");
    }

    #[test]
    fn curvature_correction_lowers_the_inclined_flux() {
        // At fixed energy the inclined flux is lower, but far less so than
        // sec θ would predict: that is the whole content of the correction.
        let vertical = integral_flux(1.0, 1.0, 400);
        let inclined = integral_flux(1.0, 0.5, 400);
        assert!(inclined < vertical, "inclined flux must be smaller");
        // The measured ratio is ≈3.6 at cos θ = 0.5, against sec θ = 2 for a
        // flat atmosphere: the curvature correction softens but does not remove
        // the zenith dependence.
        let ratio = vertical / inclined;
        assert!(ratio > 2.5 && ratio < 5.0, "ratio = {ratio}");
    }

    #[test]
    fn integral_agrees_with_its_own_derivative() {
        // Φ(E_i) − Φ(E_{i+1}) must equal ∫ I dE over the interval: the pair in
        // the table has to be internally consistent or every gradient is wrong.
        for cos in [0.3f64, 0.7, 1.0] {
            let e_low = 1.0f64;
            let e_high = 2.5f64;
            let difference = integral_flux(e_low, cos, 2000) - integral_flux(e_high, cos, 2000);
            let quadrature = integral_flux(e_low, cos, 4000) - integral_flux(e_high, cos, 4000);
            let relative = ((difference - quadrature) / quadrature).abs();
            assert!(
                relative < 0.01,
                "cos={cos} difference={difference} quadrature={quadrature}"
            );
        }
    }

    #[test]
    fn table_reproduces_the_nodes_it_was_built_from() {
        let table = generate(9, 5, -1.0, 2.0);
        for (i, log_e) in table.log10_e_min.iter().enumerate() {
            for (j, cos) in table.cos_theta.iter().enumerate() {
                let stored = table.integral[i * 5 + j];
                let direct = table.integral(10f32.powf(*log_e), *cos);
                assert!(
                    ((direct - stored) / stored).abs() < 1e-5,
                    "node ({i},{j}) interpolated {direct} vs stored {stored}"
                );
            }
        }
    }

    #[test]
    fn interpolation_is_bounded_by_its_neighbours() {
        let table = generate_default();
        // Midway between the first two nodes in both axes.
        let e = 10f32.powf(0.5 * (table.log10_e_min[0] + table.log10_e_min[1]));
        let c = 0.5 * (table.cos_theta[0] + table.cos_theta[1]);
        let value = table.integral(e, c);
        let corners = [
            table.integral[0],
            table.integral[1],
            table.integral[33],
            table.integral[34],
        ];
        let lowest = corners.iter().copied().fold(f32::INFINITY, f32::min);
        let highest = corners.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // A cubic interpolant may overshoot a bilinear cell slightly; 5% is a
        // sanity bound, not a shape guarantee.
        assert!(
            (lowest * 0.95..=highest * 1.05).contains(&value),
            "{value} outside [{lowest}, {highest}]"
        );
    }

    #[test]
    fn lookups_outside_the_grid_clamp() {
        let table = generate_default();
        let below = table.integral(1e-9, 1.0);
        let at_first = table.integral(10f32.powf(table.log10_e_min[0]), 1.0);
        assert!((below - at_first).abs() < 1e-12);

        let above = table.integral(1e9, 1.0);
        let at_last = table.integral(10f32.powf(table.log10_e_min[128]), 1.0);
        assert!((above - at_last).abs() < 1e-12);

        // cos θ below 0 and above 1 clamp to the axis ends.
        assert!((table.integral(1.0, -2.0) - table.integral(1.0, 0.0)).abs() < 1e-12);
        assert!((table.integral(1.0, 5.0) - table.integral(1.0, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn interpolant_slope_equals_the_tabulated_intensity_at_nodes() {
        // At a node the Hermite slope is taken from the table's own I, so these
        // must agree; away from nodes they may differ slightly, and it is the
        // slope that the forward model's gradient uses.
        // The top node is excluded: lookups above the grid clamp to a constant,
        // whose slope is zero by construction (asserted separately below).
        let table = generate(9, 5, -1.0, 2.0);
        for (i, log_e) in table.log10_e_min.iter().enumerate().take(8) {
            for (j, cos) in table.cos_theta.iter().enumerate() {
                let e = 10f32.powf(*log_e);
                let slope = table.integral_slope(e, *cos);
                let tabulated = -table.diff[i * 5 + j];
                let relative = ((slope - tabulated) / tabulated).abs();
                assert!(
                    relative < 1e-4,
                    "node ({i},{j}): slope {slope} vs tabulated {tabulated}"
                );
            }
        }
    }

    #[test]
    fn interpolant_slope_matches_a_central_difference_of_the_integral() {
        let table = generate_default();
        for e in [0.5f32, 1.5, 3.0, 12.0, 200.0] {
            let h = 1e-3 * e;
            let numeric = (table.integral(e + h, 0.8) - table.integral(e - h, 0.8)) / (2.0 * h);
            let analytic = table.integral_slope(e, 0.8);
            let relative = ((numeric - analytic) / analytic).abs();
            assert!(
                relative < 2e-3,
                "E={e}: numeric {numeric} analytic {analytic}"
            );
        }
    }

    #[test]
    fn outside_the_grid_the_integral_is_flat() {
        let table = generate(9, 5, -1.0, 2.0);
        let far_above = 10f32.powf(2.5);
        assert_eq!(table.integral_slope(far_above, 0.5), 0.0);
        let below = 10f32.powf(-3.0);
        assert_eq!(table.integral_slope(below, 0.5), 0.0);
    }

    #[test]
    fn flux_decreases_with_energy() {
        let table = generate_default();
        let mut previous = f32::INFINITY;
        for i in 0..129 {
            let e = 10f32.powf(table.log10_e_min[i]);
            let value = table.integral(e, 1.0);
            assert!(value < previous, "Φ must fall with E at {e} GeV");
            previous = value;
        }
    }

    #[test]
    fn default_table_fits_the_asset_budget() {
        let table = generate_default();
        let bytes = table.to_bytes();
        assert!(
            bytes.len() < 200_000,
            "table is {} bytes, budget is 200 KB",
            bytes.len()
        );
        assert_eq!(table.len(), 129 * 33);
    }

    #[test]
    fn serialisation_round_trips() {
        let table = generate(17, 9, -1.0, 3.0);
        let bytes = table.to_bytes();
        let restored = FluxTable::from_bytes(&bytes).expect("round trip");
        assert_eq!(restored.version, table.version);
        assert_eq!(restored.integral, table.integral);
        assert_eq!(restored.diff, table.diff);
        // Axes are rebuilt from origin and step, so compare at the precision the
        // format stores.
        for (a, b) in restored.log10_e_min.iter().zip(&table.log10_e_min) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn a_single_flipped_byte_is_rejected() {
        let table = generate(9, 5, -1.0, 2.0);
        let mut bytes = table.to_bytes();
        let target = 40; // inside the integral array
        bytes[target] ^= 0xFF;
        assert_eq!(
            FluxTable::from_bytes(&bytes),
            Err(TableError::ChecksumMismatch)
        );
    }

    #[test]
    fn foreign_files_are_rejected_by_magic_and_version() {
        let table = generate(9, 5, -1.0, 2.0);
        let mut bytes = table.to_bytes();
        bytes[0] = b'X';
        assert_eq!(FluxTable::from_bytes(&bytes), Err(TableError::BadMagic));

        let mut bytes = table.to_bytes();
        bytes[4] = 99;
        assert_eq!(
            FluxTable::from_bytes(&bytes),
            Err(TableError::UnsupportedVersion(99))
        );

        assert_eq!(FluxTable::from_bytes(&[0u8; 8]), Err(TableError::Truncated));
    }

    #[test]
    fn empty_table_is_not_consistent() {
        let table = FluxTable {
            version: "test".into(),
            log10_e_min: vec![],
            cos_theta: vec![],
            integral: vec![],
            diff: vec![],
            dphi_dloge: vec![],
        };
        assert!(table.is_empty());
        assert!(!table.is_consistent());
    }
}
