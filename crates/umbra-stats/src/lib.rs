//! Detection statistics: is the void real, and how sure are we?
//!
//! Home of A6 (profile-likelihood scan with look-elsewhere correction), A8
//! (sequential testing) and A11 (uncertainty). The one kernel implemented today
//! is the Asimov test statistic, because both the detection scan and the
//! experiment-design utilities are expressed in terms of it.

pub mod laplace;

pub use laplace::{
    poisson_laplace, GraphCurvature, LaplaceConfig, PoissonProblem, SigmaMap, Uncertainty,
};

/// Median expected significance of `H1` against `H0`, as a `Z` value.
///
/// Evaluates the Asimov test statistic
///
/// ```text
/// TS_A = 2 Σ_i [ λ1 ln(λ1 / λ0) − (λ1 − λ0) ]
/// ```
///
/// and returns `sqrt(TS_A)`. Because `TS_A` is linear in exposure,
/// `Z_A ∝ sqrt(t)`, which is what makes the `t_5σ = 25 t0 / TS_A(t0)` forecast
/// in the design tab possible.
///
/// Returns `0.0` when the hypotheses coincide, and clamps negative statistics
/// (which can only come from non-nested inputs) to zero.
///
/// # Panics
///
/// Panics if the two slices differ in length.
pub fn asimov_significance(lambda0: &[f32], lambda1: &[f32]) -> f32 {
    assert_eq!(
        lambda0.len(),
        lambda1.len(),
        "H0 and H1 expectations must share a ray-cell basis"
    );
    let mut ts = 0.0f64;
    for (&l0, &l1) in lambda0.iter().zip(lambda1) {
        let (l0, l1) = (f64::from(l0), f64::from(l1));
        if l0 <= 0.0 || l1 <= 0.0 {
            continue;
        }
        ts += 2.0 * (l1 * (l1 / l0).ln() - (l1 - l0));
    }
    if ts <= 0.0 {
        0.0
    } else {
        ts.sqrt() as f32
    }
}

/// Exposure needed for a median `5σ` discovery, given a current exposure.
///
/// Scales the Asimov statistic linearly with time: `t_5σ = 25 t0 / TS_A(t0)`.
/// Returns `None` when the current configuration yields no expected signal.
pub fn time_to_five_sigma(t0_seconds: f32, z_at_t0: f32) -> Option<f32> {
    let ts = f64::from(z_at_t0) * f64::from(z_at_t0);
    if ts <= 0.0 {
        return None;
    }
    Some((25.0 * f64::from(t0_seconds) / ts) as f32)
}

/// A scanned candidate: where the template sits and how significant it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanCandidate {
    /// Flat voxel index of the template centre.
    pub voxel: usize,
    /// Template scale index (multi-scale bank).
    pub scale: u8,
    /// Local test statistic.
    pub ts: f32,
    /// Local significance `Z = U / sqrt(I)`.
    pub z: f32,
    /// Best-fit amplitude in `[0, 1]`.
    pub alpha_hat: f32,
}

/// Configuration of a multi-scale detection scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanConfig {
    /// Template scales, as radii in voxels.
    pub scales_voxels: Vec<f32>,
    /// Amplitude bounds `[min, max]` enforced during profile fits.
    pub alpha_bounds: [f32; 2],
    /// Whether nuisance parameters are profiled.
    pub profile_nuisances: bool,
}

/// Calibration of the global (look-elsewhere-corrected) null distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct LookElsewhereCalibration {
    /// Number of background-only datasets simulated.
    pub n_toys: u32,
    /// Low threshold `u0` at which the Euler characteristic was measured.
    pub u0: f32,
    /// Mean Euler characteristic of the excursion set at `u0`.
    pub mean_euler_at_u0: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_hypotheses_have_no_significance() {
        let l = [10.0f32, 20.0, 5.0];
        assert_eq!(asimov_significance(&l, &l), 0.0);
    }

    #[test]
    fn a_void_that_raises_counts_is_significant() {
        let l0 = [100.0f32, 100.0, 100.0];
        let l1 = [110.0f32, 110.0, 110.0];
        let z = asimov_significance(&l0, &l1);
        assert!(z > 1.0, "expected a positive Z, got {z}");
        assert!(z.is_finite());
    }

    #[test]
    fn significance_grows_with_the_void() {
        let l0 = [100.0f32; 4];
        let small = asimov_significance(&l0, &[105.0; 4]);
        let large = asimov_significance(&l0, &[120.0; 4]);
        assert!(large > small);
    }

    #[test]
    fn time_to_five_sigma_scales_inverse_quadratically() {
        let t = time_to_five_sigma(86_400.0, 2.0).expect("non-zero signal");
        // TS = 4 at one day, so 25/4 of the time is needed.
        assert!((t - 86_400.0 * 25.0 / 4.0).abs() < 1.0, "t = {t}");
        assert!(time_to_five_sigma(86_400.0, 0.0).is_none());
    }
}
