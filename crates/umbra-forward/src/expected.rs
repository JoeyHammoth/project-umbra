//! The L0/L1 forward model with exact gradients (spec §2.3).
//!
//! ```text
//! λ_i    = T · K_i · Φ(> E_min(X_i), θ_i) + b_i
//! ∂λ_i/∂X = − T · K_i · I(E_min(X_i), θ_i) · a · e^{b X_i}
//! ```
//!
//! `K_i` is `RayCell::k_scale` (area in cm², efficiency, zenith cosine and bin
//! solid angle), so exposure only multiplies at evaluation time and the system
//! survives the exposure slider unchanged.
//!
//! The gradient is analytic, not finite-differenced: L-BFGS-B, PDHG and the
//! Fisher maps all consume it, and a difference quotient there would add noise
//! exactly where the solvers are most sensitive.

use umbra_physics::{de_min_dx, e_min, Fidelity, FluxTable, STANDARD_ROCK_A, STANDARD_ROCK_B};

use crate::system::System;

/// Physical parameters of a forward evaluation.
///
/// Defaults are the standard-rock values of spec §2.3; a scenario may override
/// them, which is why they travel as data rather than as constants at the call
/// site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardParams {
    /// CSDA ionisation term `a`, in GeV cm² g⁻¹.
    pub a: f32,
    /// CSDA radiative term `b`, in cm² g⁻¹.
    pub b: f32,
    /// L0 attenuation length `Λ`, in g cm⁻².
    pub l0_attenuation_gcm2: f32,
    /// L0 reference flux `Φ₀`, in cm⁻² s⁻¹ sr⁻¹.
    pub l0_flux0: f32,
    /// Background rate per ray cell, in Hz (applied at L2 and optionally L1).
    pub background_hz: f32,
}

impl Default for ForwardParams {
    fn default() -> Self {
        Self {
            a: STANDARD_ROCK_A,
            b: STANDARD_ROCK_B,
            l0_attenuation_gcm2: 1000.0,
            l0_flux0: 7.0e-3,
            background_hz: 0.0,
        }
    }
}

/// Expected counts and their derivative with respect to opacity.
#[derive(Debug, Clone, PartialEq)]
pub struct Expected {
    /// `λ_i` in counts.
    pub lambda: Vec<f32>,
    /// `∂λ_i/∂X_i` in counts per (g cm⁻²).
    pub dlambda_dx: Vec<f32>,
    /// `X_i = (L ρ)_i` in g cm⁻², retained because every solver wants it.
    pub opacity: Vec<f32>,
}

/// Evaluates `λ` and `∂λ/∂X` for a density field.
///
/// `exposure_s` is the observation time, `fidelity` selects the physics, and
/// `background_hz` is only used at L2 (L0 and L1 are background-free unless a
/// non-zero rate is supplied, which the scenarios do for detector noise floors).
pub fn expected(
    system: &System,
    rho: &[f32],
    flux: &FluxTable,
    fidelity: Fidelity,
    exposure_s: f32,
    params: &ForwardParams,
) -> Expected {
    let opacity = system.opacity(rho);
    let exposure = exposure_s.max(0.0);
    let mut lambda = Vec::with_capacity(opacity.len());
    let mut dlambda_dx = Vec::with_capacity(opacity.len());

    let background: f32 = match fidelity {
        Fidelity::L2 => params.background_hz * exposure,
        _ => 0.0,
    };

    for (ray, x) in system.rays.iter().zip(&opacity) {
        let k = ray.k_scale;
        let x = x.max(0.0);
        match fidelity {
            Fidelity::L0 => {
                let transmission = (-x / params.l0_attenuation_gcm2).exp();
                let value = exposure * k * params.l0_flux0 * transmission + background;
                let derivative =
                    -exposure * k * params.l0_flux0 / params.l0_attenuation_gcm2 * transmission;
                lambda.push(value);
                dlambda_dx.push(derivative);
            }
            Fidelity::L1 | Fidelity::L2 => {
                let e = e_min(x, params.a, params.b);
                let cos = ray.zen.cos();
                // Value and derivative in one interpolation. The slope comes from
                // the interpolant itself, so the gradient differentiates the same
                // model the value came from (see FluxTable::integral_and_slope).
                let (phi, dphi_de) = flux.integral_and_slope(e, cos);
                let value = exposure * k * phi + background;
                let derivative = exposure * k * dphi_de * de_min_dx(x, params.a, params.b);
                lambda.push(value);
                dlambda_dx.push(derivative);
            }
        }
    }

    Expected {
        lambda,
        dlambda_dx,
        opacity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::RayCell;
    use umbra_core::{Csr, Grid2, GridAny};
    use umbra_physics::generate_default;

    fn system_with(k_scale: f32, zen: f32, path_cm: f32, voxels: usize) -> System {
        // A single ray crossing `voxels` voxels of uniform path length.
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for j in 0..voxels {
            indices.push(j as u32);
            data.push(path_cm);
        }
        let l = Csr::new(vec![0, voxels as u32], indices, data);
        System {
            l,
            rays: vec![RayCell {
                det: 0,
                zen,
                az: 0.0,
                k_scale,
            }],
            grid: GridAny::Two(Grid2::new([voxels as u32, 1], [0.0, 0.0], 1.0)),
        }
    }

    #[test]
    fn l0_reduces_to_the_beer_lambert_form() {
        // Hand-computed: X = 500 g/cm², Λ = 1000, Φ₀ = 7e-3, K = 1e4 cm²,
        // T = 100 s.
        let system = system_with(1.0e4, 0.0, 250.0, 2);
        let rho = [1.0f32, 1.0];
        let params = ForwardParams::default();
        let result = expected(
            &system,
            &rho,
            &generate_default(),
            Fidelity::L0,
            100.0,
            &params,
        );
        let expected_value = 100.0 * 1.0e4 * 7.0e-3 * (-500.0f32 / 1000.0).exp();
        assert!(
            (result.lambda[0] - expected_value).abs() / expected_value < 1e-5,
            "lambda {} vs {}",
            result.lambda[0],
            expected_value
        );
        assert!(result.dlambda_dx[0] < 0.0, "transmission must fall with X");
    }

    #[test]
    fn transmission_is_a_ratio_to_a_threshold_not_to_zero_energy() {
        // Dividing by Φ(>0) would be meaningless: the soft end of the spectrum is
        // enormous and no detector counts it. Against a 1 GeV threshold the
        // table (verified against an independent scipy quadrature to 0.1%) gives
        // ≈70% through 3.8 m of rock and ≈1% through ~75 m. The "thick structure"
        // 1% regime in spec §2.1 is the second case; a 3.8 m wall is thin.
        let flux = generate_default();
        let ratio = |x: f32| {
            let e = e_min(x, STANDARD_ROCK_A, STANDARD_ROCK_B);
            flux.integral(e, 1.0) / flux.integral(1.0, 1.0)
        };
        let thin = ratio(1007.0);
        let thick = ratio(20_000.0);
        assert!(
            (0.60..0.80).contains(&thin),
            "3.8 m of rock transmits {thin} above 1 GeV"
        );
        assert!(
            (0.008..0.02).contains(&thick),
            "75 m of rock transmits {thick} above 1 GeV"
        );
        assert!(thin > thick, "transmission must fall with depth");
    }

    #[test]
    fn gradient_matches_a_central_difference_in_opacity() {
        // Multiply the density by (1 ± δ): X scales linearly, so the numerical
        // derivative in X isolates ∂λ/∂X.
        let flux = generate_default();
        let params = ForwardParams::default();
        let system = system_with(2.5e3, 0.3, 180.0, 3);
        let rho_base = [2.65f32, 2.65, 2.65];
        let analytic = expected(&system, &rho_base, &flux, Fidelity::L1, 1000.0, &params);

        // Counts are stored in f32, so the step has to be large enough that the
        // difference is not dominated by rounding (a 1e-4 step differs by ~0.2
        // counts out of 500, i.e. within f32 noise) and small enough that the
        // derivative has not curved: 1e-2 is the compromise.
        let delta = 1e-2f32;
        let rho_plus = rho_base.map(|r| r * (1.0 + delta));
        let rho_minus = rho_base.map(|r| r * (1.0 - delta));
        let plus = expected(&system, &rho_plus, &flux, Fidelity::L1, 1000.0, &params);
        let minus = expected(&system, &rho_minus, &flux, Fidelity::L1, 1000.0, &params);

        let dx = plus.opacity[0] - minus.opacity[0];
        let numeric = (plus.lambda[0] - minus.lambda[0]) / dx;
        let relative = ((numeric - analytic.dlambda_dx[0]) / analytic.dlambda_dx[0]).abs();
        assert!(
            relative < 5e-3,
            "numeric {numeric} analytic {} relative {relative}",
            analytic.dlambda_dx[0]
        );
    }

    #[test]
    fn zero_efficiency_and_zero_exposure_stay_finite() {
        let flux = generate_default();
        let params = ForwardParams::default();
        for (k_scale, exposure) in [(0.0f32, 100.0f32), (1.0e3, 0.0), (0.0, 0.0)] {
            let system = system_with(k_scale, 0.0, 200.0, 2);
            let result = expected(
                &system,
                &[2.65, 2.65],
                &flux,
                Fidelity::L1,
                exposure,
                &params,
            );
            assert!(result.lambda[0].is_finite(), "lambda not finite");
            assert!(result.dlambda_dx[0].is_finite(), "gradient not finite");
            assert!(result.lambda[0] >= 0.0);
        }
    }

    #[test]
    fn counts_are_linear_in_exposure() {
        let flux = generate_default();
        let params = ForwardParams::default();
        let system = system_with(1.0e4, 0.0, 400.0, 2);
        let rho = [2.0f32, 2.0];
        let a = expected(&system, &rho, &flux, Fidelity::L1, 100.0, &params);
        let b = expected(&system, &rho, &flux, Fidelity::L1, 400.0, &params);
        let ratio = b.lambda[0] / a.lambda[0];
        assert!((ratio - 4.0).abs() < 1e-4, "ratio {ratio}");
    }

    #[test]
    fn a_lower_fidelity_is_selectable_and_changes_the_scale() {
        let flux = generate_default();
        let params = ForwardParams::default();
        let system = system_with(1.0e4, 0.0, 100.0, 2);
        let rho = [2.65f32, 2.65];
        let l0 = expected(&system, &rho, &flux, Fidelity::L0, 1000.0, &params);
        let l1 = expected(&system, &rho, &flux, Fidelity::L1, 1000.0, &params);
        assert!(l0.lambda[0] > 0.0 && l1.lambda[0] > 0.0);
        assert_ne!(l0.lambda[0], l1.lambda[0], "fidelity must change the model");
    }
}
