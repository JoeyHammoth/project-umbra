//! A11 on the shared scene: every solver's estimate gets a σ map, and the map
//! says "unconstrained" where the geometry saw nothing.

mod common;

use umbra_core::Grid2;
use umbra_forward::System;
use umbra_physics::generate_default;
use umbra_physics::Fidelity;
use umbra_recon::{
    tv_curvature, A1Config, A2Config, A3Config, A4Config, Control, OpacityPdhg, PoissonMap,
    Problem, ResultCard, Solver, SolverKind, SparseDeviation, Start, TransmissionEm,
};
use umbra_stats::laplace::LaplaceConfig;

/// Runs a solver for a fixed number of steps and returns its estimate.
///
/// The card is about the error bars of an estimate, not about convergence, so a
/// short run keeps the test fast without weakening what it checks.
fn estimate<S: Solver>(problem: &Problem<'_>, cfg: &S::Config, start: &Start<'_>) -> Vec<f32> {
    let mut solver = S::init(problem, cfg, start);
    let ctl = Control::new(20, 20);
    for _ in 0..20 {
        if solver.step(&ctl).converged {
            break;
        }
    }
    solver.estimate().to_vec()
}

#[test]
fn the_shared_scene_carries_error_bars_from_every_solver() {
    let golden = common::load();
    let (system, density, counts) = common::system_and_counts();
    let counts: Vec<f32> = counts.iter().map(|count| *count as f32).collect();
    let flux = generate_default();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let scene = &golden.scene;
    let voxels = problem.voxel_count();

    // The fixture integrity check every test binary repeats: a golden that
    // drifted from its scene would make every number below describe other data.
    assert_eq!(golden.density.len(), voxels);
    assert_eq!(density.len(), voxels);
    assert_eq!(golden.counts.len(), problem.ray_count());
    assert_eq!(golden.scene.scene_key_hex(), golden.scene_key);
    assert_eq!(golden.seed, golden.scene.seed);
    assert_eq!(golden.exposure_s, golden.scene.exposure_s);
    assert_eq!(golden.lambda.len(), problem.ray_count());
    assert_eq!(
        golden.count_hash,
        umbra_forward::counts_hash(&golden.counts),
        "the counts do not hash to the recorded digest"
    );

    // A1 is an L0 method and refuses L1 data, so its card is the L0 problem on
    // the same operator and counts: a card has to describe the model the solver
    // actually fitted, not the one the data was generated under.
    let l0 = Problem::new(
        &system,
        &counts,
        &flux,
        Fidelity::L0,
        scene.exposure_s,
        scene.forward_params(),
    );
    let a1 = estimate::<TransmissionEm>(&l0, &A1Config::from_scene(scene), &Start::default());
    let a2 = estimate::<PoissonMap>(&problem, &A2Config::from_scene(scene), &Start::default());
    let a3 = estimate::<OpacityPdhg>(&problem, &A3Config::from_scene(scene), &Start::default());
    let bulk = vec![scene.physics.rho_bulk; voxels];
    let a4 = estimate::<SparseDeviation>(
        &problem,
        &A4Config::from_scene(scene),
        &Start::from_reference(&bulk),
    );

    // The set the card must call unconstrained is a geometric fact, so derive it
    // from the operator instead of trusting the card to grade itself: a voxel no
    // stored path length touches has no Fisher information at all. On this
    // 32-ray grid that set is large, which is the single-view problem in one
    // number.
    // A stored entry with zero path length is a ray that grazed the voxel's
    // corner: it carries no information, so it does not count as seeing it.
    let mut covered = vec![false; voxels];
    for row in 0..problem.ray_count() {
        for (column, value) in system.l.row(row) {
            if value > 0.0 {
                covered[column] = true;
            }
        }
    }
    let hidden = covered.iter().filter(|seen| !**seen).count();

    let cases = [
        (SolverKind::TransmissionEm, &a1, &l0),
        (SolverKind::PoissonMap, &a2, &problem),
        (SolverKind::OpacityPdhg, &a3, &problem),
        (SolverKind::SparseDeviation, &a4, &problem),
    ];

    for (kind, rho, its_problem) in cases {
        let prior = tv_curvature(its_problem, rho, 1.0e3, 0.05);
        let card = ResultCard::new(
            kind,
            its_problem,
            rho,
            Some(&prior),
            &LaplaceConfig::default(),
        );
        assert_eq!(card.sigma.len(), voxels, "{}", kind.id());
        assert!(
            card.has_error_bars(),
            "{}: {} columns unresolved, worst residual {:e} over {} iterations",
            kind.id(),
            card.sigma.unresolved,
            card.sigma.worst_residual,
            card.sigma.cg_iterations
        );
        assert_eq!(
            card.unconstrained(),
            hidden,
            "{}: the card's unconstrained set must be the operator's unseen voxels",
            kind.id()
        );
        for (voxel, seen) in covered.iter().enumerate() {
            assert_eq!(
                card.sigma.at(voxel).is_constrained(),
                *seen,
                "{}: voxel {voxel} constrained={} but covered={seen}",
                kind.id(),
                card.sigma.at(voxel).is_constrained()
            );
        }
        if hidden > 0 {
            assert!(card.unconstrained() < voxels, "a grid with no bars at all");
        }
        // Every voxel the geometry actually sees carries a finite, positive bar.
        for (voxel, seen) in covered.iter().enumerate() {
            if !seen {
                continue;
            }
            let sigma = card
                .sigma
                .at(voxel)
                .sigma()
                .expect("a covered voxel must have a bar");
            assert!(
                sigma.is_finite() && sigma > 0.0,
                "{}: voxel {voxel} has sigma {sigma}",
                kind.id()
            );
        }
        // A bar wider than the whole box would be a sign the prior curvature is
        // missing rather than a sign the scene is hard: on this grid the data
        // may not pin a voxel, but the prior at least keeps σ bounded by the
        // density range.
        let worst = card.sigma.worst_sigma().expect("constrained voxels");
        assert!(
            worst < scene.physics.rho_bulk * 10.0,
            "{}: worst sigma {worst} is not a bar, it is an unregularised null direction",
            kind.id()
        );
    }
}

#[test]
fn a_voxel_no_ray_touches_is_reported_unconstrained_on_its_card() {
    // Three voxels, two rays: the third is invisible to the geometry. The card
    // must say so rather than report a σ the prior invented.
    let system = System {
        l: umbra_core::Csr::new(vec![0, 1, 2], vec![0, 1], vec![200.0f32, 200.0]),
        rays: (0..2)
            .map(|_| umbra_forward::RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0e4,
            })
            .collect(),
        grid: umbra_core::GridAny::Two(Grid2::new([3, 1], [0.0, 0.0], 0.1)),
        // No scene here: the fixture writes its own operator, so it has no poses
        // to carry and `detector_pose` answers `None`.
        poses: Vec::new(),
    };
    let flux = generate_default();
    let counts = [40.0f32, 40.0];
    let problem = Problem::new(
        &system,
        &counts,
        &flux,
        umbra_physics::Fidelity::L0,
        1.0,
        umbra_forward::ForwardParams::default(),
    );
    let estimate = [2.65f32, 2.65, 2.65];
    let card = ResultCard::new(
        SolverKind::PoissonMap,
        &problem,
        &estimate,
        None,
        &LaplaceConfig::default(),
    );
    assert_eq!(card.unconstrained(), 1);
    assert_eq!(card.sigma.at(2).sigma(), None);
    assert!(card.sigma.at(0).sigma().is_some());
    assert!(card.sigma.at(1).sigma().is_some());
}

#[test]
fn the_prior_keeps_a_limited_angle_grid_from_having_no_bars() {
    // With one view the data Fisher matrix is singular; the TV curvature is what
    // makes the posterior precision invertible. Removing the prior must show up
    // as unresolved columns rather than as smaller σ.
    let golden = common::load();
    let (system, _, counts) = common::system_and_counts();
    let counts: Vec<f32> = counts.iter().map(|count| *count as f32).collect();
    let flux = generate_default();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let rho = vec![golden.scene.physics.rho_bulk; problem.voxel_count()];

    let with_prior = tv_curvature(&problem, &rho, 1.0e3, 0.05);
    let card = ResultCard::new(
        SolverKind::PoissonMap,
        &problem,
        &rho,
        Some(&with_prior),
        &LaplaceConfig::default(),
    );
    assert!(
        card.has_error_bars(),
        "the regularised solve must converge ({} unresolved, worst residual {:e})",
        card.sigma.unresolved,
        card.sigma.worst_residual
    );
    assert!(card.sigma.worst_sigma().is_some());

    // The bare solve is asked with a small budget on purpose: the point is the
    // verdict, not how long it takes to fail.
    let bare_budget = LaplaceConfig {
        tolerance: 1.0e-7,
        max_cg_iters: 40,
        // No prior at all — no TV curvature and no level: the data-only matrix
        // is singular, which is the state this test is asking about.
        level_sigma: 0.0,
    };
    let bare = ResultCard::new(SolverKind::PoissonMap, &problem, &rho, None, &bare_budget);
    assert!(
        !bare.has_error_bars(),
        "32 rays cannot constrain 1024 voxels; the data-only solve must not claim it did"
    );
}
