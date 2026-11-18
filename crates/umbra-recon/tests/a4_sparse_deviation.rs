//! A4: the one-sided proximal step, the box the deviation never leaves, the
//! monotone objective, and the corridor that A2's smoothness prior leaves
//! half-filled.

mod common;

use umbra_forward::scene::SceneSpec;
use umbra_forward::{build_system_2d, expected};
use umbra_physics::{generate_default, Fidelity};
use umbra_recon::{
    prox, A2Config, A4Config, Control, DeviationSign, PoissonMap, Problem, Solver, SparseDeviation,
};

const MULTIVIEW: &str = "web/src/scenes/multiview.json";
const MULTIVIEW_GOLDEN: &str = "crates/umbra-recon/tests/golden/multiview.json";

/// The minimiser of `½(δ−z)² + step·μ|δ|` on the feasible interval, by ternary
/// search. The function is convex, so this is the definition rather than a
/// numerical approximation of it.
fn minimiser(sign: DeviationSign, reference: f32, z: f64, step: f64, mu: f64) -> f64 {
    let (mut lo, mut hi) = match sign {
        DeviationSign::Void => (-f64::from(reference), 0.0),
        DeviationSign::Inclusion => (0.0, f64::from(reference)),
    };
    let cost = |delta: f64| 0.5 * (delta - z) * (delta - z) + step * mu * delta.abs();
    for _ in 0..200 {
        let third = (hi - lo) / 3.0;
        let left = lo + third;
        let right = hi - third;
        if cost(left) < cost(right) {
            hi = right;
        } else {
            lo = left;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn the_datasets_are_the_ones_the_committed_scenes_describe() {
    // Every field of the fixture, from both scenes: a golden that drifted from
    // its scene would make every measurement below describe the wrong data.
    for (scene_path, golden_path) in [
        (common::SCENE_PATH, common::GOLDEN_PATH),
        (MULTIVIEW, MULTIVIEW_GOLDEN),
    ] {
        let golden = common::load_from(scene_path, golden_path);
        let system = build_system_2d(&golden.scene.to_scene_2d());
        assert_eq!(
            golden.count_hash,
            umbra_forward::counts_hash(&golden.counts),
            "{golden_path}: the counts do not hash to the recorded digest"
        );
        assert_eq!(golden.scene.scene_key_hex(), golden.scene_key);
        assert_eq!(golden.seed, golden.scene.seed);
        assert_eq!(golden.exposure_s, golden.scene.exposure_s);
        assert_eq!(golden.lambda.len(), system.ray_count());
        assert_eq!(golden.counts.len(), system.ray_count());
        assert_eq!(golden.density.len(), system.voxel_count());
    }

    let (_, density, counts) = common::system_and_counts();
    let golden = common::load();
    assert_eq!(density, golden.density);
    assert_eq!(counts, golden.counts);
}

#[test]
fn the_prox_is_the_minimiser_of_its_own_definition() {
    // Random inputs, including the two bounds: a closed form that is only right
    // in the interior is the failure mode worth catching, since the box is where
    // most of the voxels end up.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f64) / ((1u64 << 24) as f64)
    };

    let mut worst: f64 = 0.0;
    for _ in 0..200 {
        for sign in [DeviationSign::Void, DeviationSign::Inclusion] {
            let reference = 0.5 + 3.0 * uniform();
            let z = -4.0 + 8.0 * uniform();
            let step = 0.01 + 2.0 * uniform();
            let mu = 20.0 * uniform();
            let closed = prox(sign, reference as f32, z, step, mu, reference as f32);
            let expected = minimiser(sign, reference as f32, z, step, mu);
            worst = worst.max((closed - expected).abs());
        }
    }
    println!("worst prox disagreement with the definition: {worst:.3e}");
    assert!(
        worst < 1.0e-6,
        "the closed-form prox is {worst:.3e} away from the minimiser"
    );

    // The definition at the boundaries, where the closed form has to clamp rather
    // than shift. Compared to the f32 reference the clamp uses, not to a decimal
    // literal, so the check is about the clamp and not about `2.65` in binary.
    let reference = 2.65f32;
    let lower = -f64::from(reference);
    assert_eq!(
        prox(DeviationSign::Void, reference, 5.0, 1.0, 1.0, 4.0),
        0.0
    );
    assert_eq!(
        prox(DeviationSign::Void, reference, -50.0, 1.0, 1.0, 4.0),
        lower
    );
    assert_eq!(
        prox(DeviationSign::Inclusion, reference, -50.0, 1.0, 1.0, 4.0),
        0.0
    );
    assert_eq!(
        prox(DeviationSign::Inclusion, reference, 50.0, 1.0, 1.0, 4.0),
        4.0
    );
}

#[test]
fn the_deviation_never_leaves_the_one_sided_box() {
    let golden = common::load_from(MULTIVIEW, MULTIVIEW_GOLDEN);
    let system = build_system_2d(&golden.scene.to_scene_2d());
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let reference = vec![golden.scene.physics.rho_bulk; problem.voxel_count()];

    let mut solver = SparseDeviation::new(&problem, A4Config::default(), &reference);
    let ctl = Control::new(200, 1);
    for step in 0..60 {
        solver.step(&ctl);
        for (voxel, (deviation, reference)) in solver.deviation().iter().zip(&reference).enumerate()
        {
            assert!(
                deviation.is_finite() && *deviation <= 0.0 && *deviation >= -reference,
                "step {step}, voxel {voxel}: δ = {deviation} outside [−{reference}, 0]"
            );
            // The estimate is the physical density, so this also says it stays
            // non-negative without a separate clamp.
            assert!(solver.estimate()[voxel] >= 0.0);
        }
    }
}

#[test]
fn the_objective_never_rises_and_converges_in_under_two_hundred_steps() {
    let golden = common::load();
    let system = build_system_2d(&golden.scene.to_scene_2d());
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let reference = vec![golden.scene.physics.rho_bulk; problem.voxel_count()];

    let mut solver =
        SparseDeviation::new(&problem, A4Config::from_scene(&golden.scene), &reference);
    let ctl = Control::new(200, 1);
    let mut seen = vec![solver.objective()];
    let mut converged = false;
    for _ in 0..200 {
        let report = solver.step(&ctl);
        seen.push(report.objective.expect("A4 reports its objective"));
        if report.converged {
            converged = true;
            break;
        }
    }

    for (step, pair) in seen.windows(2).enumerate() {
        assert!(
            pair[1] <= pair[0] + 1.0e-12 * pair[0].abs(),
            "the objective rose at step {step}: {} -> {}",
            pair[0],
            pair[1]
        );
    }
    println!(
        "A4 on the shared scene: {} steps, objective {:.6e} -> {:.6e}, L = {:.3e}, \
         nll {:.6e} + l1 {:.3e}",
        solver.iterations(),
        seen[0],
        solver.objective(),
        solver.lipschitz(),
        solver.nll(),
        solver.l1()
    );
    assert!(converged, "A4 did not converge in 200 steps");
    assert!(solver.iterations() < 200, "A4 needed the whole budget");
    assert!(
        solver.objective() < seen[0],
        "the objective did not fall at all"
    );
}

/// The void's own footprint from the scene's geometry.
///
/// Not "wherever the truth is below bulk": the slabs are 2 m thick in a 16 m
/// grid, so that mask is dominated by the vacuum outside the material and
/// measures the background rather than the cavity.
fn footprint(scene: &SceneSpec, half_length: f32, radius: f32) -> Vec<bool> {
    let nx = scene.grid.n[0] as usize;
    let h = scene.grid.h_m;
    (0..scene.grid.cell_count())
        .map(|voxel| {
            let i = (voxel % nx) as f32;
            let j = (voxel / nx) as f32;
            let x = scene.grid.origin[0] + (i + 0.5) * h;
            let y = scene.grid.origin[1] + (j + 0.5) * h;
            x.abs() <= half_length && y.abs() <= radius
        })
        .collect()
}

/// Mean `|estimate − truth|` over a mask.
fn error(estimate: &[f32], truth: &[f32], mask: &[bool]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for ((model, truth), inside) in estimate.iter().zip(truth).zip(mask) {
        if *inside {
            total += (f64::from(*model) - f64::from(*truth)).abs();
            count += 1;
        }
    }
    total / count.max(1) as f64
}

#[test]
fn a4_recovers_the_corridor_where_a2_leaves_it_half_filled() {
    // The single-view scene is where the prior matters: 32 rays through one
    // direction, so the corridor is a null-space direction for a method that
    // only wants smoothness. A4's one-sided bound says the corridor cannot be
    // denser than the rock around it, which is information TV does not have.
    let golden = common::load();
    let system = build_system_2d(&golden.scene.to_scene_2d());
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let reference = vec![golden.scene.physics.rho_bulk; problem.voxel_count()];
    let truth = &golden.density;
    let mask = footprint(&golden.scene, 3.5, 0.5);
    assert!(mask.iter().filter(|inside| **inside).count() > 20);

    let ctl = Control::new(400, 1);
    let mut a2 = PoissonMap::new(&problem, A2Config::from_scene(&golden.scene));
    for _ in 0..400 {
        if a2.step(&ctl).converged {
            break;
        }
    }
    let mut a4 = SparseDeviation::new(&problem, A4Config::from_scene(&golden.scene), &reference);
    for _ in 0..400 {
        if a4.step(&ctl).converged {
            break;
        }
    }

    let (a2_error, a4_error) = (
        error(a2.estimate(), truth, &mask),
        error(a4.estimate(), truth, &mask),
    );
    let (a2_fit, a4_fit) = (problem.nll(a2.estimate()), problem.nll(a4.estimate()));
    println!(
        "corridor: A4 {a4_error:.3} g cm⁻³ ({} steps) vs A2 {a2_error:.3} ({} steps); \
         fits {a4_fit:.6e} vs {a2_fit:.6e}",
        a4.iterations(),
        a2.iterations()
    );

    assert!(
        a4_error < 0.8 * a2_error,
        "A4 left {a4_error:.3} g cm⁻³ in the corridor against A2's {a2_error:.3}"
    );
    // The win has to be a prior doing its job, not a worse fit being called a
    // better reconstruction.
    assert!(
        a4_fit <= a2_fit + 0.01 * a2_fit.abs(),
        "A4's fit {a4_fit:.6e} is far from A2's {a2_fit:.6e}"
    );
}

#[test]
fn four_views_supply_the_information_the_prior_was_standing_in_for() {
    // The honest boundary of A4. On the same amount of rock seen from four
    // sides, TV already localises the void and the one-sided prior has nothing
    // left to add; the ordering reverses. Recorded rather than hidden, because
    // a portfolio whose members each win somewhere is more useful than one whose
    // favourite wins everywhere.
    let golden = common::load_from(MULTIVIEW, MULTIVIEW_GOLDEN);
    let system = build_system_2d(&golden.scene.to_scene_2d());
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let reference = vec![golden.scene.physics.rho_bulk; problem.voxel_count()];
    let mask = footprint(&golden.scene, 2.1, 0.6);

    let ctl = Control::new(400, 1);
    let mut a2 = PoissonMap::new(&problem, A2Config::from_scene(&golden.scene));
    for _ in 0..400 {
        if a2.step(&ctl).converged {
            break;
        }
    }
    let mut a4 = SparseDeviation::new(&problem, A4Config::from_scene(&golden.scene), &reference);
    for _ in 0..400 {
        if a4.step(&ctl).converged {
            break;
        }
    }

    let (a2_error, a4_error) = (
        error(a2.estimate(), &golden.density, &mask),
        error(a4.estimate(), &golden.density, &mask),
    );
    println!("four views: A4 {a4_error:.3} g cm⁻³, A2 {a2_error:.3}");
    assert!(
        a4_error < 0.5,
        "A4 did not recover the buried void at all: {a4_error:.3} g cm⁻³"
    );
}

#[test]
fn dense_inclusion_mode_recovers_a_block_added_to_the_reference() {
    // The sign flip: the same machinery with `δ ≥ 0` finds material the
    // reference does not contain, which is the granite-block case. Solids add in
    // this scene format, so the block primitive carries 1.4 g cm⁻³ on top of the
    // rubble's 2.0.
    let scene = SceneSpec::from_json_str(INCLUSION_SCENE).expect("the synthetic scene loads");
    let system = build_system_2d(&scene.to_scene_2d());
    let flux = generate_default();
    let params = scene.forward_params();
    let truth = scene.density();
    let counts = expected(
        &system,
        &truth,
        &flux,
        Fidelity::L0,
        scene.exposure_s,
        &params,
    )
    .lambda;
    let problem = Problem::new(
        &system,
        &counts,
        &flux,
        Fidelity::L0,
        scene.exposure_s,
        params,
    );

    let bulk = scene.physics.rho_bulk;
    let reference = vec![bulk; problem.voxel_count()];
    let block: Vec<bool> = truth.iter().map(|rho| *rho > bulk + 0.3).collect();
    assert!(
        block.iter().filter(|inside| **inside).count() > 20,
        "the block is too small to measure"
    );
    for (voxel, inside) in block.iter().enumerate() {
        assert!(
            !*inside || truth[voxel] > bulk,
            "voxel {voxel} is marked as block but is not denser than the reference"
        );
    }

    let cfg = A4Config {
        sign: DeviationSign::Inclusion,
        ..A4Config::from_scene(&scene)
    };
    let mut solver = SparseDeviation::new(&problem, cfg.clone(), &reference);
    let ctl = Control::new(400, 1);
    for _ in 0..400 {
        if solver.step(&ctl).converged {
            break;
        }
    }

    let (mut recovered, mut inside) = (0.0, 0usize);
    let (mut outside, mut outside_count) = (0.0, 0usize);
    for (voxel, deviation) in solver.deviation().iter().enumerate() {
        assert!(
            *deviation >= 0.0 && *deviation <= cfg.delta_max,
            "voxel {voxel}: δ = {deviation} outside [0, {}]",
            cfg.delta_max
        );
        if block[voxel] {
            recovered += f64::from(*deviation);
            inside += 1;
        } else {
            outside += f64::from(deviation.abs());
            outside_count += 1;
        }
    }
    let recovered = recovered / inside as f64;
    let outside = outside / outside_count as f64;
    println!(
        "inclusion: block mean δ {recovered:.3} g cm⁻³ (truth 1.400), elsewhere mean |δ| \
         {outside:.3}, {} steps",
        solver.iterations()
    );
    assert!(
        recovered > 0.5,
        "the block was not recovered: mean δ {recovered:.3} g cm⁻³"
    );
    assert!(
        outside < 0.3,
        "the inclusion mode invented material elsewhere: mean |δ| {outside:.3}"
    );
}

/// A 16×16 slice of 2.0 g cm⁻³ rubble with a 3.4 g cm⁻³ block inside it, seen
/// from four sides. The outer box fills the grid so the reference is the whole
/// scene rather than a slab floating in vacuum.
const INCLUSION_SCENE: &str = r#"{
  "schema": "umbra.scene/1",
  "id": "a4-inclusion",
  "label": "Granite block in rubble",
  "units": { "length": "m", "density": "g/cm3" },
  "primitives": [
    { "kind": "box", "c": [0, 0, 0], "half": [4, 4, 4], "rot": [0, 0, 0, 1], "rho": 2.0 },
    { "kind": "box", "c": [0, 0, 0], "half": [1.5, 4, 1.0], "rot": [0, 0, 0, 1], "rho": 1.4 }
  ],
  "voids": [],
  "detectors": [
    { "pose": { "p": [0, -6, 0], "n": [0, 1, 0], "up": [1, 0, 0] }, "area_m2": 0.5, "eff": 0.9,
      "bins": { "flat": { "min": -0.5, "max": 0.5, "bins": 12 } }, "sigma_ang_mrad": 5, "bg_rate_hz": 0.0 },
    { "pose": { "p": [0, 6, 0], "n": [0, -1, 0], "up": [1, 0, 0] }, "area_m2": 0.5, "eff": 0.9,
      "bins": { "flat": { "min": -0.5, "max": 0.5, "bins": 12 } }, "sigma_ang_mrad": 5, "bg_rate_hz": 0.0 },
    { "pose": { "p": [-6, 0, 0], "n": [1, 0, 0], "up": [0, 1, 0] }, "area_m2": 0.5, "eff": 0.9,
      "bins": { "flat": { "min": -0.5, "max": 0.5, "bins": 12 } }, "sigma_ang_mrad": 5, "bg_rate_hz": 0.0 },
    { "pose": { "p": [6, 0, 0], "n": [-1, 0, 0], "up": [0, 1, 0] }, "area_m2": 0.5, "eff": 0.9,
      "bins": { "flat": { "min": -0.5, "max": 0.5, "bins": 12 } }, "sigma_ang_mrad": 5, "bg_rate_hz": 0.0 }
  ],
  "physics": { "level": "L0", "a": 2.0e-3, "b": 4.0e-6, "rho_bulk": 2.0, "x0": 26.5 },
  "exposure_s": 2000000,
  "grid": { "n": [16, 16], "h_m": 0.5, "origin_m": [-4, -4] },
  "seed": 7,
  "notes": "A 3 m granite block inside 2.0 g cm-3 rubble: the case A4's inclusion mode exists for, where the deviation may only add material."
}"#;
