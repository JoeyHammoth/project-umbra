//! Continuous slowing-down approximation (CSDA) range–energy relation.
//!
//! With `-dE/dX = a + b E` the minimum surface energy needed to traverse
//! opacity `X` is
//!
//! ```text
//! E_min(X) = (a / b) * (exp(b X) - 1)
//! ```
//!
//! and its derivative, which drives every gradient-based method, is
//!
//! ```text
//! dE_min/dX = a * exp(b X)
//! ```

/// Minimum surface energy, in GeV, to cross opacity `x_gcm2` (g cm⁻²).
///
/// `a` is in GeV cm² g⁻¹ and `b` in cm² g⁻¹. Uses `exp_m1` so thin targets keep
/// their precision.
pub fn e_min(x_gcm2: f32, a: f32, b: f32) -> f32 {
    (a / b) * (b * x_gcm2).exp_m1()
}

/// Derivative `dE_min/dX`, in GeV per (g cm⁻²).
pub fn de_min_dx(x_gcm2: f32, a: f32, b: f32) -> f32 {
    a * (b * x_gcm2).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{STANDARD_ROCK_A, STANDARD_ROCK_B};

    #[test]
    fn zero_opacity_needs_no_energy() {
        assert_eq!(e_min(0.0, STANDARD_ROCK_A, STANDARD_ROCK_B), 0.0);
        assert_eq!(de_min_dx(0.0, STANDARD_ROCK_A, STANDARD_ROCK_B), STANDARD_ROCK_A);
    }

    #[test]
    fn matched_opacity_is_about_two_gev() {
        // 1 m of standard rock is ~265 g/cm², 3.8 m is ~1000 g/cm².
        let e = e_min(1000.0, STANDARD_ROCK_A, STANDARD_ROCK_B);
        assert!((e - 2.004).abs() < 0.01, "E_min = {e} GeV");
    }

    #[test]
    fn e_min_is_monotone() {
        let mut previous = 0.0;
        for step in 1..=50 {
            let e = e_min(step as f32 * 50.0, STANDARD_ROCK_A, STANDARD_ROCK_B);
            assert!(e > previous, "E_min must increase with opacity");
            previous = e;
        }
    }

    #[test]
    fn derivative_matches_central_difference() {
        let (a, b) = (STANDARD_ROCK_A, STANDARD_ROCK_B);
        for x in [10.0f32, 250.0, 1000.0, 4000.0] {
            let h = 1e-2 * x.max(1.0);
            let numeric = (e_min(x + h, a, b) - e_min(x - h, a, b)) / (2.0 * h);
            let analytic = de_min_dx(x, a, b);
            let rel = ((numeric - analytic) / analytic).abs();
            assert!(rel < 1e-3, "x={x} numeric={numeric} analytic={analytic}");
        }
    }
}
