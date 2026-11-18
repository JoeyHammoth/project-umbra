//! The second scene: four detectors, a buried void, and the ordered-subset
//! criterion the single-view default scene cannot carry.
//!
//! The default scene has one detector, so its 32 rays are one view and subsets
//! partition a single direction. Here four poses give four views, which is the
//! situation ordered subsets were designed for, and the void sits at the
//! mid-plane rather than breaking a surface, which is the situation a
//! smoothness prior flattens and A4's one-sided anchor is for.

mod common;

use umbra_forward::{build_system_2d, counts_hash, expected};
use umbra_physics::{generate_default, Fidelity};
use umbra_recon::{A1Config, Control, Problem, Solver, TransmissionEm};

const SCENE: &str = "web/src/scenes/multiview.json";
const GOLDEN: &str = "crates/umbra-recon/tests/golden/multiview.json";

#[test]
fn the_four_pose_dataset_is_the_one_the_committed_scene_describes() {
    // The default fixture through both accessors, so the loader and the
    // convenience call cannot drift apart unnoticed.
    let default = common::load();
    let (_, _, default_counts) = common::system_and_counts();
    assert_eq!(default_counts, default.counts);
    assert_eq!(default.count_hash, counts_hash(&default.counts));
    assert_eq!(default.seed, default.scene.seed);
    assert_eq!(default.exposure_s, default.scene.exposure_s);
    assert_eq!(default.density.len(), default.scene.grid.cell_count());
    assert_eq!(default.lambda.len(), default.counts.len());

    let multiview = common::load_from(SCENE, GOLDEN);
    assert_ne!(
        multiview.scene_key, default.scene_key,
        "the second scene must be a different dataset from the first"
    );
    assert_eq!(multiview.scene.scene_key_hex(), multiview.scene_key);
    assert_eq!(multiview.seed, multiview.scene.seed);
    assert_eq!(multiview.count_hash, counts_hash(&multiview.counts));

    let system = build_system_2d(&multiview.scene.to_scene_2d());
    assert_eq!(
        system.ray_count(),
        64,
        "four detectors with sixteen bins each"
    );
    assert_eq!(multiview.counts.len(), system.ray_count());
    let mut detectors: Vec<u16> = system.rays.iter().map(|ray| ray.det).collect();
    detectors.sort_unstable();
    detectors.dedup();
    assert_eq!(detectors.len(), 4, "four distinct detector poses");

    // The density is derived from the scene, not a second copy of it.
    for (index, (derived, stored)) in multiview
        .scene
        .density()
        .iter()
        .zip(&multiview.density)
        .enumerate()
    {
        assert!(
            (derived - stored).abs() <= 1e-6,
            "voxel {index}: scene says {derived} g/cm3, golden says {stored}"
        );
    }

    // And λ reproduces from the density, so a physics change cannot hide behind
    // a committed count vector.
    let flux = generate_default();
    let model = expected(
        &system,
        &multiview.density,
        &flux,
        multiview.scene.physics.level,
        multiview.exposure_s,
        &multiview.scene.forward_params(),
    );
    for (index, (fresh, stored)) in model.lambda.iter().zip(&multiview.lambda).enumerate() {
        let relative = (fresh - stored).abs() / stored.abs().max(1e-6);
        assert!(
            relative <= 1e-5,
            "ray {index}: lambda {fresh} vs golden {stored} (relative {relative:.2e})"
        );
    }
}

/// The objective at a solver's own starting point.
fn initial_objective(problem: &Problem<'_>, cfg: A1Config) -> f64 {
    TransmissionEm::new(problem, cfg).objective()
}

/// Steps A1 takes to reach `target`, and the objective it reached.
fn steps_to_target(problem: &Problem<'_>, cfg: A1Config, target: f64) -> (u32, f64) {
    let mut solver = TransmissionEm::new(problem, cfg);
    let ctl = Control::new(5_000, 1);
    for _ in 0..5_000 {
        let report = solver.step(&ctl);
        if solver.objective() <= target || report.converged {
            break;
        }
    }
    (solver.iterations(), solver.objective())
}

#[test]
fn ordered_subsets_reach_the_same_objective_in_fewer_sweeps() {
    // The criterion re-scoped out of #10, finally measurable: a full sweep is
    // one step, an ordered-subset sweep is `subsets` steps, and the comparison
    // is of sweeps — the thing that costs time — to the same objective.
    //
    // The scene is evaluated at L0 because A1 is an L0 method and this test is
    // about iteration counts, not about which physics is right.
    let golden = common::load_from(SCENE, GOLDEN);
    let mut scene = golden.scene.clone();
    scene.physics.level = Fidelity::L0;

    let system = build_system_2d(&scene.to_scene_2d());
    let flux = generate_default();
    let counts = expected(
        &system,
        &golden.density,
        &flux,
        Fidelity::L0,
        scene.exposure_s,
        &scene.forward_params(),
    )
    .lambda;
    let problem = Problem::new(
        &system,
        &counts,
        &flux,
        Fidelity::L0,
        scene.exposure_s,
        scene.forward_params(),
    );

    let base = A1Config::from_scene(&scene);
    let full_cfg = A1Config {
        subsets: 1,
        ..base.clone()
    };
    let start = initial_objective(&problem, full_cfg.clone());
    let (full_steps, full_objective) = steps_to_target(&problem, full_cfg, f64::NEG_INFINITY);
    let full_sweeps = f64::from(full_steps);

    // Compare at 90% of the decrease a full sweep achieves. Every setting can
    // reach that; the last few parts in 10⁶ are exactly where a guarded subset
    // step stops descending and reports convergence instead, so requiring
    // agreement there would be measuring the stopping rule rather than the
    // acceleration.
    let target = start + 0.9 * (full_objective - start);
    let slack = 1.0e-5 * target.abs();
    println!("start {start:.6e}, full {full_objective:.6e} in {full_steps} sweeps, 90% target {target:.6e}");

    // The number of subsets is a tuning choice, and the criterion is about what
    // the method can do, so measure the range rather than assuming four.
    let mut best_sweeps = f64::INFINITY;
    let mut best_subsets = 0usize;
    for subsets in [2usize, 4, 8, 16] {
        let (steps, objective) = steps_to_target(
            &problem,
            A1Config {
                subsets: subsets as u32,
                ..base.clone()
            },
            target + slack,
        );
        let sweeps = f64::from(steps) / subsets as f64;
        println!(
            "subsets {subsets:>2}: {steps} steps = {sweeps:.2} sweeps, objective {objective:.6e}"
        );
        assert!(
            objective <= target + slack,
            "{subsets} subsets did not reach the 90% objective: {objective:.6e} vs {target:.6e}"
        );
        if sweeps < best_sweeps {
            best_sweeps = sweeps;
            best_subsets = subsets;
        }
    }

    println!(
        "full: {full_steps} sweeps to the 90% objective; best {best_subsets} subsets: \
         {best_sweeps:.2} sweeps ({:.2}x)",
        best_sweeps / full_sweeps
    );
    assert!(
        best_sweeps <= 0.5 * full_sweeps,
        "ordered subsets took {best_sweeps:.2} sweeps against {full_sweeps:.0}: not the half the \
         criterion asks for"
    );
}
