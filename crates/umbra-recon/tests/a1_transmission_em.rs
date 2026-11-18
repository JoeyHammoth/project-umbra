//! A1 against the shared default scene and against a known three-voxel void.
//!
//! The committed golden dataset is L1 and A1 is an L0 method, so the accuracy
//! claims here use the *same committed scene* evaluated at L0 — same geometry,
//! density, detectors, exposure and seed, through [`common::load`] — and the L1
//! golden counts are used only to show the solver stays bounded on data its
//! model does not describe. Both paths read the committed fixtures; neither
//! invents its own.

mod common;

use umbra_core::Grid2;
use umbra_forward::{build_system_2d, expected, Scene2D, System};
use umbra_physics::{generate_default, AngularBinning, Detector, Fidelity, Pose};
use umbra_recon::{A1Config, Control, Solver, TransmissionEm};

/// The committed scene at L0, with counts equal to its own expectation.
fn default_l0() -> (System, Vec<f32>, Vec<f32>, A1Config) {
    let golden = common::load();
    let system = build_system_2d(&golden.scene.to_scene_2d());

    let mut scene = golden.scene.clone();
    scene.physics.level = Fidelity::L0;
    let model = expected(
        &system,
        &golden.density,
        &generate_default(),
        Fidelity::L0,
        scene.exposure_s,
        &scene.forward_params(),
    );

    let cfg = A1Config {
        subsets: 8,
        ..A1Config::from_scene(&scene)
    };
    (system, model.lambda, golden.density, cfg)
}

/// Worst per-ray relative difference between the model and the data.
fn misfit(solver: &TransmissionEm, counts: &[f32]) -> f64 {
    solver
        .lambda()
        .iter()
        .zip(counts)
        .map(|(model, data)| (f64::from(*model) - f64::from(*data)).abs() / f64::from(*data))
        .fold(0.0f64, f64::max)
}

/// Runs at most `steps` steps, returning the objective after each.
fn objectives(system: &System, counts: &[f32], cfg: &A1Config, steps: u32) -> Vec<f64> {
    let mut solver = TransmissionEm::new(system, counts, cfg.clone());
    let ctl = Control::new(steps, 1);
    let mut seen = Vec::new();
    for _ in 0..steps {
        let report = solver.step(&ctl);
        seen.push(report.objective.expect("A1 has an objective"));
        if report.converged {
            break;
        }
    }
    seen
}

#[test]
fn the_objective_never_increases_on_the_default_scene() {
    let (system, counts, _, cfg) = default_l0();
    let mut solver = TransmissionEm::new(&system, &counts, cfg.clone());
    let start = solver.objective();
    let ctl = Control::new(50, 1);
    let mut seen = vec![start];
    for _ in 0..50 {
        seen.push(solver.step(&ctl).objective.expect("A1 has an objective"));
    }

    for (step, pair) in seen.windows(2).enumerate() {
        assert!(
            pair[1] <= pair[0],
            "objective rose at step {step}: {} -> {}",
            pair[0],
            pair[1]
        );
    }
    // NLL is negative here, so "less than half" would be satisfied by any
    // decrease at all; compare the size of the decrease to the objective scale.
    let decrease = seen[0] - seen[49];
    assert!(
        decrease > 0.05 * seen[0].abs(),
        "50 steps moved the NLL from {} to {}: {decrease:.3e} is too little",
        seen[0],
        seen[49]
    );
}

#[test]
fn the_fit_improves_steadily_on_the_default_scene() {
    let (system, counts, _, cfg) = default_l0();
    let mut solver = TransmissionEm::new(&system, &counts, cfg);
    let initial = misfit(&solver, &counts);
    let ctl = Control::new(200, 1);
    for _ in 0..200 {
        if solver.step(&ctl).converged {
            break;
        }
    }
    let final_misfit = misfit(&solver, &counts);

    // An exact fit is not claimed: with 32 ray cells against 1024 voxels the
    // problem is massively underdetermined, and the estimate is projected at
    // zero density over most of the grid, so the solver settles at a constrained
    // point. What it must do is improve a lot from the uniform start.
    assert!(
        final_misfit < 0.25,
        "worst ray is {final_misfit:.3} off after the run, from {initial:.3}"
    );
    assert!(
        final_misfit < initial / 4.0,
        "the fit barely improved: {initial:.3} -> {final_misfit:.3}"
    );

    for (voxel, rho) in solver.estimate().iter().enumerate() {
        assert!(
            rho.is_finite() && *rho >= 0.0 && *rho <= 4.0,
            "voxel {voxel} left the box: {rho}"
        );
    }
}

#[test]
fn ordered_subsets_reach_the_same_solution() {
    // The subset partition must not change the problem, only the path to it:
    // after enough steps both settings must land on the same objective.
    let (system, counts, _, cfg) = default_l0();
    let mut full = TransmissionEm::new(
        &system,
        &counts,
        A1Config {
            subsets: 1,
            ..cfg.clone()
        },
    );
    let mut subsets = TransmissionEm::new(
        &system,
        &counts,
        A1Config {
            subsets: 4,
            ..cfg.clone()
        },
    );
    let ctl = Control::new(400, 1);
    for _ in 0..400 {
        if full.step(&ctl).converged {
            break;
        }
    }
    for _ in 0..400 {
        if subsets.step(&ctl).converged {
            break;
        }
    }
    let relative = (full.objective() - subsets.objective()).abs() / full.objective().abs();
    println!(
        "subsets: full {:.1} in {} steps, 4 subsets {:.1} in {} steps, relative {relative:.2e}",
        full.objective(),
        full.iterations(),
        subsets.objective(),
        subsets.iterations()
    );
    assert!(
        relative < 0.01,
        "the two settings disagree by {relative:.2e} of the objective: full {:.1}, subsets {:.1}",
        full.objective(),
        subsets.objective()
    );
}

#[test]
fn it_recovers_a_three_voxel_void_from_noiseless_counts() {
    // A 4 m × 1.5 m block at 0.5 m with one empty column: the void is three
    // voxels, and it is the only thing the data can explain, because every ray
    // through that column loses all of its attenuation while its neighbours pin
    // the bulk. A void buried inside the block would be ambiguous under this
    // limited-angle geometry, and no unregularised method could be held to 10%
    // there — that ambiguity is what A4's prior exists for.
    let grid = Grid2::new([8, 3], [-2.0, -0.75], 0.5);
    let detector = |x: f32| Detector {
        pose: Pose {
            position: [x, -6.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            up: [1.0, 0.0, 0.0],
        },
        area_m2: 1.0,
        eff: 0.9,
        binning: AngularBinning::Flat2D {
            min: -0.5,
            max: 0.5,
            bins: 8,
        },
        sigma_ang_mrad: 0.0,
        bg_rate_hz: 0.0,
    };
    let system = build_system_2d(&Scene2D::new(
        grid.clone(),
        vec![detector(-1.5), detector(-0.5), detector(0.5), detector(1.5)],
    ));

    let bulk = 2.65f32;
    let mut truth = vec![bulk; grid.cell_count()];
    let mut void = Vec::new();
    for row in 0..3 {
        let index = grid.index(3, row).expect("inside the grid");
        truth[index] = 0.0;
        void.push(index);
    }

    let exposure = 1.0e6f32;
    let model = expected(
        &system,
        &truth,
        &generate_default(),
        Fidelity::L0,
        exposure,
        &umbra_forward::ForwardParams::default(),
    );
    let cfg = A1Config {
        exposure_s: exposure,
        rho_init: bulk,
        rho_max: 4.0,
        ..A1Config::default()
    };

    let mut solver = TransmissionEm::new(&system, &model.lambda, cfg);
    let ctl = Control::new(400, 1);
    for _ in 0..400 {
        if solver.step(&ctl).converged {
            break;
        }
    }

    let estimate = solver.estimate();
    let void_mean: f32 = void.iter().map(|&index| estimate[index]).sum::<f32>() / void.len() as f32;
    assert!(
        void_mean <= 0.1 * bulk,
        "void recovered at {void_mean:.3} g/cm3, more than 10% of the {bulk} g/cm3 bulk"
    );

    let bulk_indices: Vec<usize> = (0..grid.cell_count())
        .filter(|i| !void.contains(i))
        .collect();
    let bulk_mean: f32 = bulk_indices
        .iter()
        .map(|&index| estimate[index])
        .sum::<f32>()
        / bulk_indices.len() as f32;
    assert!(
        (bulk_mean - bulk).abs() <= 0.1 * bulk,
        "bulk recovered at {bulk_mean:.3} g/cm3, more than 10% from {bulk}"
    );
}

#[test]
fn cancellation_between_subsets_leaves_the_estimate_where_it_was() {
    let (system, counts, _, cfg) = default_l0();
    let mut solver = TransmissionEm::new(&system, &counts, cfg);
    let ctl = Control::new(50, 1);
    solver.step(&ctl);
    let before: Vec<f32> = solver.estimate().to_vec();
    let objective = solver.objective();

    ctl.request_stop();
    let report = solver.step(&ctl);
    assert_eq!(solver.estimate(), before.as_slice());
    assert_eq!(report.objective, Some(objective));
    assert!(!report.converged, "a cancelled step is not convergence");
}

#[test]
fn the_committed_golden_dataset_stays_stable_under_the_l0_model() {
    // The golden counts are L1 and A1 fits an L0 model, so no accuracy is
    // claimed here: a biased answer is the correct behaviour of a method used
    // outside its model. What is required is that it stays bounded, finite and
    // monotone rather than diverging when handed data it does not describe.
    let golden = common::load();
    assert_eq!(golden.scene.scene_key_hex(), golden.scene_key);
    assert_eq!(
        golden.count_hash,
        umbra_forward::counts_hash(&golden.counts)
    );
    assert_eq!(golden.seed, golden.scene.seed);
    assert_eq!(golden.exposure_s, golden.scene.exposure_s);

    let (system, _, counts) = common::system_and_counts();
    assert_eq!(golden.lambda.len(), system.ray_count());
    let counts: Vec<f32> = counts.iter().map(|count| *count as f32).collect();
    let cfg = A1Config {
        subsets: 8,
        ..A1Config::from_scene(&golden.scene)
    };

    let seen = objectives(&system, &counts, &cfg, 200);
    for (step, pair) in seen.windows(2).enumerate() {
        assert!(
            pair[1] <= pair[0],
            "objective rose at step {step}: {pair:?}"
        );
    }

    let mut solver = TransmissionEm::new(&system, &counts, cfg.clone());
    let ctl = Control::new(200, 1);
    for _ in 0..200 {
        if solver.step(&ctl).converged {
            break;
        }
    }
    assert_eq!(solver.estimate().len(), system.voxel_count());
    for (voxel, rho) in solver.estimate().iter().enumerate() {
        assert!(
            rho.is_finite() && *rho >= 0.0 && *rho <= cfg.rho_max,
            "voxel {voxel} left the box on mismatched data: {rho}"
        );
    }
}
