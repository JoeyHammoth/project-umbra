//! Parametric Bayesian inversion of a void (algorithm A7).
//!
//! Instead of reconstructing a density volume, this crate infers the parameters
//! of a void primitive directly. The forward model is an analytic chord length:
//! for ray `i`, `X_i(θ) = X_i^ref − ρ_stone · ℓ_i(θ)`, where `ℓ_i` comes from a
//! closed-form ray–quadric intersection. One likelihood evaluation therefore
//! costs `O(rays)` flops with no re-voxelisation.
//!
//! Samplers (affine-invariant ensemble MCMC, nested sampling) land with their
//! roadmap ticket; this file fixes the parameterisation and the result shape so
//! the rest of the workspace can compile against it.

/// Parameters of an ellipsoidal void.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoidParams {
    /// Centre, in metres.
    pub c: [f32; 3],
    /// Semi-axes, in metres. Ordering is a prior, not a constraint.
    pub semi: [f32; 3],
    /// Orientation, as Z-Y-X Euler angles in radians.
    pub angles: [f32; 3],
}

impl VoidParams {
    /// Volume, in m³, assuming a genuine ellipsoid.
    pub fn volume_m3(&self) -> f32 {
        let [a, b, c] = self.semi;
        (4.0 / 3.0) * std::f32::consts::PI * a * b * c
    }
}

/// Prior bounds for one parameter, uniform in the stated coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: f32,
    pub max: f32,
}

impl Bounds {
    /// Whether `value` lies inside the closed interval.
    pub fn contains(&self, value: f32) -> bool {
        value >= self.min && value <= self.max
    }

    /// Width of the interval.
    pub fn span(&self) -> f32 {
        self.max - self.min
    }
}

/// The priors the sampler draws from, derived from the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct VoidPrior {
    /// Bounds on each centre coordinate.
    pub centre: [Bounds; 3],
    /// Bounds on each semi-axis.
    pub semi: [Bounds; 3],
    /// Include "no void" as a model, giving a Bayes factor rather than a
    /// posterior conditional on a void existing.
    pub with_null_model: bool,
}

/// Posterior summaries rendered in the characterisation card.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorSummary {
    /// Median and 16th/84th percentile of each centre coordinate.
    pub centre: [Bounds; 3],
    /// Median and 16th/84th percentile of each semi-axis.
    pub semi: [Bounds; 3],
    /// Log evidence of the void model, when nested sampling has run.
    pub log_evidence: Option<f32>,
    /// Log Bayes factor against the no-void model.
    pub log_bayes_factor: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipsoid_volume_matches_sphere_formula() {
        let params = VoidParams {
            c: [0.0; 3],
            semi: [1.0; 3],
            angles: [0.0; 3],
        };
        let expected = 4.0 / 3.0 * std::f32::consts::PI;
        assert!((params.volume_m3() - expected).abs() < 1e-5);
    }

    #[test]
    fn bounds_behave_as_closed_intervals() {
        let b = Bounds { min: 0.0, max: 2.0 };
        assert!(b.contains(0.0));
        assert!(b.contains(2.0));
        assert!(!b.contains(2.5));
        assert_eq!(b.span(), 2.0);
    }
}
