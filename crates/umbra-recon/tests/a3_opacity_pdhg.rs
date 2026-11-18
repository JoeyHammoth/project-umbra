//! A3: the per-ray inversion that replaces the likelihood, the operator norm
//! that sets the step sizes, the dual certificate, and where A3 agrees with A2.

mod common;

use umbra_core::Grid2;
use umbra_forward::expected::ray_lambda;
use umbra_forward::{build_system_2d, ForwardParams, Scene2D, System};
use umbra_physics::{generate_default, AngularBinning, Detector, Fidelity, FluxTable, Pose};
use umbra_recon::{A2Config, A3Config, Control, OpacityPdhg, PoissonMap, Problem, Solver};

/// A tiny L1 scene: a 4×4 grid and three detectors.
struct Fixture {
    system: System,
    flux: FluxTable,
    params: ForwardParams,
    exposure_s: f32,
    nx: usize,
    ny: usize,
}

impl Fixture {
    fn problem<'a>(&'a self, counts: &'a [f32]) -> Problem<'a> {
        Problem::new(
            &self.system,
            counts,
            &self.flux,
            Fidelity::L1,
            self.exposure_s,
            self.params,
        )
    }
}

fn small() -> Fixture {
    let grid = Grid2::new([4, 4], [-1.0, -1.0], 0.5);
    let detector = |x: f32| Detector {
        pose: Pose {
            position: [x, -6.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            up: [1.0, 0.0, 0.0],
        },
        area_m2: 1.0,
        eff: 0.9,
        binning: AngularBinning::Flat2D {
            min: -0.4,
            max: 0.4,
            bins: 4,
        },
        sigma_ang_mrad: 0.0,
        bg_rate_hz: 0.0,
    };
    let system = build_system_2d(&Scene2D::new(
        grid.clone(),
        vec![detector(-1.0), detector(0.0), detector(1.0)],
    ));
    assert_eq!(system.ray_count(), 12);
    Fixture {
        system,
        flux: generate_default(),
        params: ForwardParams::default(),
        exposure_s: 1.0e6,
        nx: 4,
        ny: 4,
    }
}

/// `‖K‖₂` computed from an explicitly assembled dense `[L; ∇]`, independently of
/// the sparse power iteration the solver uses.
fn dense_norm(system: &System, nx: usize, ny: usize, iterations: u32) -> f64 {
    let voxels = nx * ny;
    let rays = system.ray_count();
    let rows = rays + 2 * voxels;
    let mut k = vec![vec![0.0f64; voxels]; rows];
    for (ray, row) in k.iter_mut().enumerate().take(rays) {
        for (voxel, length) in system.l.row(ray) {
            row[voxel] = f64::from(length);
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let here = i + nx * j;
            if i + 1 < nx {
                k[rays + here][here] = -1.0;
                k[rays + here][here + 1] = 1.0;
            }
            if j + 1 < ny {
                k[rays + voxels + here][here] = -1.0;
                k[rays + voxels + here][here + nx] = 1.0;
            }
        }
    }

    let mut vector = vec![1.0f64; voxels];
    let mut norm = 0.0;
    for _ in 0..iterations {
        // Kᵀ(K v), densely.
        let mut forward = vec![0.0f64; rows];
        for (row, out) in k.iter().zip(forward.iter_mut()) {
            *out = row.iter().zip(&vector).map(|(a, b)| a * b).sum();
        }
        let mut back = vec![0.0f64; voxels];
        for (row, value) in k.iter().zip(&forward) {
            for (voxel, entry) in row.iter().enumerate() {
                back[voxel] += entry * value;
            }
        }
        norm = vector
            .iter()
            .zip(&back)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .sqrt();
        let size = back.iter().map(|value| value * value).sum::<f64>().sqrt();
        if size <= 0.0 {
            return 0.0;
        }
        for (value, next) in vector.iter_mut().zip(&back) {
            *value = next / size;
        }
    }
    norm
}

#[test]
fn the_inversion_handles_counts_from_zero_to_ten_thousand() {
    // The ticket's range, and the two ends are where a Newton solve actually
    // misbehaves: at zero the root is at infinity, at 10⁴ the f32 count has
    // eleven significant bits and the model is a smooth function of a
    // logarithmically-spaced argument.
    let fixture = small();
    let counts: Vec<f32> = vec![0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]
        .into_iter()
        .cycle()
        .take(fixture.system.ray_count())
        .collect();
    let problem = fixture.problem(&counts);
    let solver = OpacityPdhg::new(&problem, A3Config::default());
    let opacity = solver.opacity_estimate();
    assert_eq!(opacity.len(), counts.len());

    let mut worst = 0.0f64;
    for (ray, x) in opacity.iter().enumerate() {
        let target = if counts[ray] > 0.0 {
            f64::from(counts[ray])
        } else {
            0.5
        };
        let (lambda, _) = ray_lambda(
            &fixture.system.rays[ray],
            *x,
            &fixture.flux,
            Fidelity::L1,
            fixture.exposure_s,
            &fixture.params,
        );
        let error = (f64::from(lambda) - target).abs() / target;
        worst = worst.max(error);
        assert!(
            error < 1.0e-3,
            "ray {ray} with {} counts inverted to X = {x}, where the model predicts {lambda}",
            counts[ray]
        );
    }
    println!("worst relative inversion error {worst:.3e}");
}

#[test]
fn a_zero_count_gives_the_finite_opacity_where_the_model_predicts_half_a_count() {
    let fixture = small();
    let counts = vec![0.0f32; fixture.system.ray_count()];
    let problem = fixture.problem(&counts);
    let solver = OpacityPdhg::new(&problem, A3Config::default());
    for (ray, x) in solver.opacity_estimate().iter().enumerate() {
        assert!(x.is_finite() && *x >= 0.0, "ray {ray} inverted to {x}");
        // Finite, but far: the correction sits near the top of what the data
        // supports rather than at the starting point.
        assert!(
            *x > 100.0,
            "a zero count should imply a deep opacity, got {x}"
        );
    }
}

#[test]
fn the_power_iteration_agrees_with_a_dense_computation() {
    let fixture = small();
    let counts = vec![100.0f32; fixture.system.ray_count()];
    let problem = fixture.problem(&counts);
    let solver = OpacityPdhg::new(&problem, A3Config::default());

    let dense = dense_norm(&fixture.system, fixture.nx, fixture.ny, 200);
    let sparse = f64::from(solver.norm_k());
    let relative = (sparse - dense).abs() / dense;
    println!("‖K‖: sparse {sparse:.6}, dense {dense:.6}, relative {relative:.2e}");
    assert!(
        relative < 1.0e-3,
        "the sparse operator and the assembled one disagree: {sparse:.6} vs {dense:.6}"
    );

    // The step sizes have to satisfy the PDHG condition with the *estimate*, so
    // this is the quantity that keeps the iteration stable.
    let (tau, sigma) = solver.steps();
    let product = tau * sigma * dense * dense;
    println!("στ‖K‖² = {product:.4}");
    assert!(product < 1.0, "step sizes violate στι‖K‖² < 1: {product}");
}

#[test]
fn the_gap_falls_and_the_primal_objective_ends_far_below_its_start() {
    let golden = common::load();
    let (system, _density, raw_counts) = common::system_and_counts();
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();

    // Check the fixture before drawing conclusions from it: a golden that drifted
    // from its scene would make every measurement below describe the wrong data.
    assert_eq!(
        golden.count_hash,
        umbra_forward::counts_hash(&golden.counts)
    );
    assert_eq!(golden.scene.scene_key_hex(), golden.scene_key);
    assert_eq!(golden.seed, golden.scene.seed);
    assert_eq!(golden.exposure_s, golden.scene.exposure_s);
    assert_eq!(golden.lambda.len(), system.ray_count());
    assert_eq!(golden.counts, raw_counts);

    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    assert_eq!(golden.density.len(), problem.voxel_count());

    let cfg = A3Config::from_scene(&golden.scene);
    let mut solver = OpacityPdhg::new(&problem, cfg.clone());
    let ctl = Control::new(200, 1);

    let mut primal = vec![solver.primal()];
    let mut gaps = vec![solver.gap()];
    for _ in 0..200 {
        let report = solver.step(&ctl);
        primal.push(report.objective.expect("A3 has a primal objective"));
        gaps.push(report.gap.expect("A3 reports its certificate"));
    }

    // PDHG is not a descent method: the primal objective can rise while the
    // dual catches up, and it does here. The guarantee the algorithm has is on
    // the *gap*, which is what this test checks; the rise is counted and printed
    // so the claim in the docs is the measured one rather than the hoped-for one.
    let rises = primal.windows(2).filter(|pair| pair[1] > pair[0]).count();
    let worst_rise = primal
        .windows(2)
        .filter(|pair| pair[1] > pair[0])
        .map(|pair| pair[1] / pair[0])
        .fold(1.0f64, f64::max);
    println!("primal rises {rises} times in 200, worst by a factor {worst_rise:.3}");

    println!(
        "primal {:.6e} -> {:.6e}; gap {:.3e} -> {:.3e} (ratio {:.3e})",
        primal[0],
        primal[primal.len() - 1],
        gaps[0],
        gaps[gaps.len() - 1],
        solver.gap_ratio()
    );
    assert!(
        solver.gap() >= 0.0,
        "a negative gap is a broken certificate"
    );
    assert!(
        gaps[gaps.len() - 1] < gaps[0],
        "the certificate did not improve: {:.3e} -> {:.3e}",
        gaps[0],
        gaps[gaps.len() - 1]
    );
    assert!(
        primal[primal.len() - 1] < primal[0],
        "the objective did not fall at all"
    );

    for (voxel, rho) in solver.estimate().iter().enumerate() {
        assert!(
            rho.is_finite() && *rho >= 0.0 && *rho <= cfg.rho_max,
            "voxel {voxel} left the box: {rho}"
        );
    }
}

#[test]
fn a3_and_a2_agree_in_structure_on_the_shared_scene() {
    let golden = common::load();
    let system = build_system_2d(&golden.scene.to_scene_2d());
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();
    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);

    let mut a2 = PoissonMap::new(&problem, A2Config::from_scene(&golden.scene));
    let mut a3 = OpacityPdhg::new(&problem, A3Config::from_scene(&golden.scene));
    let ctl = Control::new(400, 1);
    for _ in 0..400 {
        if a2.step(&ctl).converged {
            break;
        }
    }
    for _ in 0..400 {
        if a3.step(&ctl).converged {
            break;
        }
    }

    let (left, right) = (a2.estimate(), a3.estimate());
    let n = left.len() as f64;
    let mean_left = left.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let mean_right = right.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance_left = 0.0;
    let mut variance_right = 0.0;
    let mut difference = 0.0;
    for (a, b) in left.iter().zip(right) {
        let (a, b) = (f64::from(*a) - mean_left, f64::from(*b) - mean_right);
        covariance += a * b;
        variance_left += a * a;
        variance_right += b * b;
        difference += (a - b).abs();
    }
    let correlation = covariance / (variance_left.sqrt() * variance_right.sqrt());

    println!(
        "A2 vs A3: correlation {correlation:.4}, mean |Δρ| {:.4} g cm⁻³, \
         fits {:.6e} vs {:.6e}",
        difference / n,
        problem.nll(left),
        problem.nll(right)
    );
    // Not an exact match, and it should not be: A2's prior is a smoothed TV and
    // A3's is the exact `‖·‖_{2,1}` at a different weight, on 32 rays for 1024
    // voxels. Where the data says nothing the two are free to differ, so the
    // agreement is structural — the corridor is in the same place — and the
    // measured 0.70 with a mean |Δρ| near 1 g cm⁻³ is the documented amount.
    // A3 reaches the truth's own fit, so the difference is the null space, not
    // accuracy.
    assert!(
        correlation > 0.6,
        "the two methods found different objects: correlation {correlation:.3}"
    );
    assert!(
        difference / n < 1.5,
        "mean |Δρ| {:.3} g cm⁻³ on a 2.65 bulk is not a disagreement, it is a different scene",
        difference / n
    );
}

#[test]
fn cancellation_leaves_the_estimate_where_it_was() {
    let fixture = small();
    let counts = vec![200.0f32; fixture.system.ray_count()];
    let problem = fixture.problem(&counts);
    let mut solver = OpacityPdhg::new(&problem, A3Config::default());
    let ctl = Control::new(50, 1);
    solver.step(&ctl);
    let before = solver.estimate().to_vec();
    let primal = solver.primal();

    ctl.request_stop();
    let report = solver.step(&ctl);
    assert_eq!(solver.estimate(), before.as_slice());
    assert_eq!(report.objective, Some(primal));
}
