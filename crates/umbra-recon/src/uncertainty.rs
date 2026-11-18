//! A11 on the result card: per-voxel error bars for a reconstruction.
//!
//! The covariance itself lives in `umbra-stats`, because A11 in M3 calibrates
//! the same object and building it twice is the obvious way for the card's bars
//! and the calibration's bars to disagree. This module is the bridge: it hands
//! the observation to the kernel and builds the prior curvature from the TV
//! term the solver actually used.
//!
//! # The prior is part of the answer
//!
//! At 32 rays on a 1024-voxel grid the data constrain at most 32 directions, so
//! the data-only Fisher matrix is singular and most of what an error bar says
//! comes from the prior. That is not a defect of the approximation — it is the
//! honest statement that a single view does not measure 992 directions — but it
//! does mean the bars must be computed with *the same* `β` and `ε` the estimate
//! was produced under. A card that used a different prior would be describing a
//! posterior nobody sampled.

use umbra_core::GridAny;
use umbra_stats::laplace::{
    poisson_laplace, GraphCurvature, LaplaceConfig, PoissonProblem, SigmaMap,
};

use crate::{Problem, SolverKind};

/// One result card: an estimate, its error bars, and which solver made it.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultCard {
    /// The solver that produced the estimate, so the card can label itself.
    pub solver: SolverKind,
    /// The density estimate the bars belong to, in g cm⁻³.
    pub estimate: Vec<f32>,
    /// Per-voxel uncertainty, in the same voxel order as the estimate.
    pub sigma: SigmaMap,
}

impl ResultCard {
    /// Computes the card for an estimate.
    ///
    /// `prior` must be the curvature of the prior the estimate was produced
    /// under; see [`tv_curvature`]. `None` is the data-only Fisher information,
    /// which is what a least-squares comparison wants and which is singular on
    /// every limited-angle scene here.
    ///
    /// # Panics
    ///
    /// Panics if `estimate` does not have one entry per voxel.
    pub fn new(
        solver: SolverKind,
        problem: &Problem<'_>,
        estimate: &[f32],
        prior: Option<&GraphCurvature>,
        cfg: &LaplaceConfig,
    ) -> Self {
        assert_eq!(
            estimate.len(),
            problem.voxel_count(),
            "the estimate must have one entry per voxel"
        );
        let observation = PoissonProblem {
            system: problem.system,
            flux: problem.flux,
            fidelity: problem.fidelity,
            exposure_s: problem.exposure_s,
            params: problem.params,
        };
        Self {
            solver,
            estimate: estimate.to_vec(),
            sigma: poisson_laplace(&observation, estimate, prior, cfg),
        }
    }

    /// Voxels the geometry cannot see at all.
    pub fn unconstrained(&self) -> usize {
        self.sigma.unconstrained()
    }

    /// Whether the bars came from a solve that reached its tolerance.
    ///
    /// A card whose bars did not converge should show the estimate without them,
    /// which is why this is a question the UI asks rather than an assertion the
    /// kernel makes.
    pub fn has_error_bars(&self) -> bool {
        self.sigma.is_converged()
    }
}

/// The TV prior's curvature at an estimate.
///
/// `TV_ε` is `d²/(2ε)` on an edge whose density difference is inside `ε` and
/// linear outside it, so its second derivative is `1/ε` inside and `0` outside:
/// edges that are already a jump carry no curvature, and edges inside the noise
/// level are penalised quadratically. That is exactly the operator a Laplace
/// approximation needs, and it is why the same `huber_eps` has to be used for
/// the estimate and for its bars.
///
/// # Panics
///
/// Panics if the grid is not two-dimensional, if the estimate does not have one
/// entry per voxel, or if `huber_eps` is not positive.
pub fn tv_curvature(
    problem: &Problem<'_>,
    rho: &[f32],
    beta: f32,
    huber_eps: f32,
) -> GraphCurvature {
    assert!(
        huber_eps.is_finite() && huber_eps > 0.0,
        "the Huber scale must be positive"
    );
    assert!(
        beta.is_finite() && beta >= 0.0,
        "the TV weight must be non-negative"
    );
    let (nx, ny) = match &problem.system.grid {
        GridAny::Two(grid) => (grid.n[0] as usize, grid.n[1] as usize),
        _ => panic!("A2's TV term is two-dimensional; a 3-D prior needs a 3-D difference"),
    };
    assert_eq!(
        rho.len(),
        problem.voxel_count(),
        "the estimate must have one entry per voxel"
    );

    let mut curvature = GraphCurvature::new(rho.len());
    if beta == 0.0 {
        return curvature;
    }
    let weight = f64::from(beta) / f64::from(huber_eps);
    for j in 0..ny {
        for i in 0..nx {
            let here = i + nx * j;
            if i + 1 < nx {
                curvature.add_edge(here, here + 1, weight);
            }
            if j + 1 < ny {
                curvature.add_edge(here, here + nx, weight);
            }
        }
    }
    curvature
}
