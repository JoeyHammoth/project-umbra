//! A2: the gradient against finite differences, convergence on the shared
//! dataset, and the L1 model where A1's Beer–Lambert law does not hold.

mod common;

use umbra_core::Grid2;
use umbra_forward::{build_system_2d, expected, ForwardParams, Scene2D, System};
use umbra_physics::{generate_default, AngularBinning, Detector, Fidelity, FluxTable, Pose};
use umbra_recon::{
    A1Config, A2Config, Control, PoissonMap, Problem, Solver, Start, TransmissionEm,
};

/// A small multi-view scene at L1, used for the finite-difference check.
struct Fixture {
    system: System,
    counts: Vec<f32>,
    flux: FluxTable,
    params: ForwardParams,
    exposure_s: f32,
    cfg: A2Config,
    nx: usize,
    ny: usize,
}

impl Fixture {
    fn problem(&self) -> Problem<'_> {
        Problem::new(
            &self.system,
            &self.counts,
            &self.flux,
            Fidelity::L1,
            self.exposure_s,
            self.params,
        )
    }
}

fn small() -> Fixture {
    let grid = Grid2::new([6, 4], [-1.5, -1.0], 0.5);
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
            bins: 6,
        },
        sigma_ang_mrad: 0.0,
        bg_rate_hz: 0.0,
    };
    let system = build_system_2d(&Scene2D::new(
        grid.clone(),
        vec![detector(-1.0), detector(0.0), detector(1.0)],
    ));

    let mut truth = vec![2.65f32; grid.cell_count()];
    truth[grid.index(2, 1).expect("inside the grid")] = 0.4;
    truth[grid.index(3, 2).expect("inside the grid")] = 1.1;

    let flux = generate_default();
    let params = ForwardParams::default();
    let exposure_s = 1.0e6;
    let counts = expected(&system, &truth, &flux, Fidelity::L1, exposure_s, &params).lambda;
    Fixture {
        system,
        counts,
        flux,
        params,
        exposure_s,
        cfg: A2Config {
            beta: 20.0,
            huber_eps: 0.05,
            rho_init: 2.65,
            rho_max: 4.0,
            ..A2Config::default()
        },
        nx: 6,
        ny: 4,
    }
}

/// `TV_ε` written straight from the definition, independently of the solver.
fn tv_reference(rho: &[f32], nx: usize, ny: usize, eps: f64) -> f64 {
    let term = |difference: f64| {
        if difference.abs() <= eps {
            difference * difference / (2.0 * eps)
        } else {
            difference.abs() - 0.5 * eps
        }
    };
    let mut total = 0.0;
    for j in 0..ny {
        for i in 0..nx {
            let here = i + nx * j;
            if i + 1 < nx {
                total += term(f64::from(rho[here + 1]) - f64::from(rho[here]));
            }
            if j + 1 < ny {
                total += term(f64::from(rho[here + nx]) - f64::from(rho[here]));
            }
        }
    }
    total
}

fn objective(problem: &Problem<'_>, rho: &[f32], fixture: &Fixture) -> f64 {
    let eps = f64::from(fixture.cfg.huber_eps);
    problem.nll(rho) + f64::from(fixture.cfg.beta) * tv_reference(rho, fixture.nx, fixture.ny, eps)
}

/// Worst per-ray relative difference between the model and the data.
fn misfit(solver: &PoissonMap, counts: &[f32]) -> f64 {
    solver
        .lambda()
        .iter()
        .zip(counts)
        .map(|(model, data)| (f64::from(*model) - f64::from(*data)).abs() / f64::from(*data))
        .fold(0.0f64, f64::max)
}
/// Worst relative disagreement between the analytic gradient and a central
/// difference of the objective, over twenty deterministic probe points.
fn worst_gradient_error(fixture: &Fixture, h: f32) -> f64 {
    let problem = fixture.problem();

    // A deterministic LCG: a test that fails on one point should fail on the
    // same point tomorrow.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };

    let voxels = problem.voxel_count();
    let mut probes: Vec<Vec<f32>> = Vec::with_capacity(20);

    // Smooth fields: every edge difference sits well inside the Huber quadratic
    // branch, where the TV term is exactly quadratic and a central difference
    // carries no truncation error at all.
    for _ in 0..10 {
        probes.push(
            (0..voxels)
                .map(|_| 2.0 + 0.02 * (uniform() - 0.5))
                .collect(),
        );
    }

    // Checkerboards: every non-zero edge is well above `ε`, where the term is
    // exactly linear. Both families deliberately avoid `|d| ≈ ε` — the single
    // place TV_ε is only C¹, where a difference quotient answers a question
    // about the kink instead of about the gradient.
    for _ in 0..10 {
        probes.push(
            (0..voxels)
                .map(|voxel| {
                    let i = voxel % fixture.nx;
                    let j = voxel / fixture.nx;
                    if (i + j).is_multiple_of(2) {
                        1.0
                    } else {
                        3.0
                    }
                })
                .collect(),
        );
    }

    let mut worst = 0.0f64;
    for rho in probes {
        let solver = PoissonMap::with_start(&problem, fixture.cfg.clone(), &Start::from_rho(&rho));
        let analytic = solver.gradient().to_vec();

        for voxel in 0..rho.len() {
            let mut plus = rho.clone();
            plus[voxel] += h;
            let mut minus = rho.clone();
            minus[voxel] -= h;
            let numeric = (objective(&problem, &plus, fixture)
                - objective(&problem, &minus, fixture))
                / f64::from(2.0 * h);
            let scale = f64::from(analytic[voxel].abs()).max(numeric.abs()).max(1.0);
            worst = worst.max((numeric - f64::from(analytic[voxel])).abs() / scale);
        }
    }
    worst
}

#[test]
fn the_gradient_matches_central_differences_at_twenty_random_points() {
    // The obvious test — "relative error below 1e-4" — is below the noise floor
    // here and would either fail forever or be met by a tolerance so loose it
    // proves nothing. `λ` is f32 and the NLL is of order 10⁸, so a difference
    // quotient with step `h` carries a relative error of about
    // `ε_f32·|NLL| / (h·|g|)`: the *measurement* is noisy, and its noise grows as
    // `1/h`.
    //
    // A genuine gradient error is a bias: it does not care what `h` is. So the
    // test measures the disagreement at two step sizes. Halving `h` must roughly
    // double the disagreement if it is noise, and must leave it unchanged if the
    // analytic gradient is simply wrong. That distinction is worth more than a
    // magic tolerance.
    let fixture = small();
    let coarse = worst_gradient_error(&fixture, 1.0e-2);
    let fine = worst_gradient_error(&fixture, 5.0e-3);
    println!("worst relative gradient error: h=1e-2 {coarse:.3e}, h=5e-3 {fine:.3e}");

    assert!(
        coarse < 1.0e-2,
        "worst relative gradient error {coarse:.3e} at h=1e-2 is too large to be noise"
    );
    assert!(
        fine > 1.4 * coarse,
        "halving the step barely changed the disagreement ({coarse:.3e} -> {fine:.3e}): \
         that is a biased gradient, not a noisy measurement"
    );
}

#[test]
fn it_converges_to_a_kkt_point_inside_the_box_on_the_shared_scene() {
    let golden = common::load();
    let system = build_system_2d(&golden.scene.to_scene_2d());
    let flux = generate_default();
    let counts: Vec<f32> = golden.counts.iter().map(|count| *count as f32).collect();

    // The fixture is the dataset every solver sees; check it before using it.
    assert_eq!(
        golden.count_hash,
        umbra_forward::counts_hash(&golden.counts)
    );
    assert_eq!(golden.scene.scene_key_hex(), golden.scene_key);
    assert_eq!(golden.seed, golden.scene.seed);
    assert_eq!(golden.lambda.len(), system.ray_count());

    let problem = Problem::from_scene(&system, &counts, &golden.scene, &flux);
    let cfg = A2Config::from_scene(&golden.scene);
    let truth_fit = problem.nll(&golden.density);

    let mut solver = PoissonMap::new(&problem, cfg.clone());
    let ctl = Control::new(600, 1);
    let mut seen = vec![solver.objective()];
    let mut converged = false;
    for _ in 0..600 {
        let report = solver.step(&ctl);
        seen.push(report.objective.expect("A2 has an objective"));
        if report.converged {
            converged = true;
            break;
        }
    }

    for (step, pair) in seen.windows(2).enumerate() {
        assert!(
            pair[1] <= pair[0] + 1.0e-12 * pair[0].abs(),
            "objective rose at step {step}: {} -> {}",
            pair[0],
            pair[1]
        );
    }
    assert!(
        converged,
        "no convergence after {} steps: KKT ratio {:.3e}",
        solver.iterations(),
        solver.kkt_ratio()
    );
    assert!(
        solver.kkt_ratio() <= f64::from(cfg.tolerance),
        "KKT ratio {:.3e} exceeds the tolerance {}",
        solver.kkt_ratio(),
        cfg.tolerance
    );
    assert!(
        seen.last().unwrap() < seen.first().unwrap(),
        "the objective did not fall at all"
    );

    for (voxel, rho) in solver.estimate().iter().enumerate() {
        assert!(
            rho.is_finite() && *rho >= 0.0 && *rho <= cfg.rho_max,
            "voxel {voxel} left the box: {rho}"
        );
    }

    let final_misfit = misfit(&solver, &counts);
    assert!(
        final_misfit < 0.2,
        "worst ray is {final_misfit:.3} off after convergence"
    );

    // The counts were sampled from the truth, so the truth is a good fit. A2
    // has to land within a few percent of it, not merely below where it started:
    // with a TV weight large enough there is always a smoother answer that fits
    // worse.
    println!("fit: truth {truth_fit:.6e}, A2 {:.6e}", solver.nll());
    assert!(
        solver.nll() - truth_fit <= 0.05 * truth_fit.abs(),
        "A2's fit {:.6e} is far from the truth's {truth_fit:.6e}",
        solver.nll()
    );
}

#[test]
fn the_l1_model_fits_l1_data_better_than_the_l0_model_does() {
    // The golden counts were generated at L1. A2 uses the flux law that made
    // them; A1 uses Beer–Lambert. Both are scored by the same L1 likelihood, so
    // the comparison is of reconstructions, not of objectives.
    let golden = common::load();
    let (system, _, raw_counts) = common::system_and_counts();
    let flux = generate_default();
    let counts: Vec<f32> = raw_counts.iter().map(|count| *count as f32).collect();
    let l1 = Problem::from_scene(&system, &counts, &golden.scene, &flux);

    let mut a2 = PoissonMap::new(&l1, A2Config::from_scene(&golden.scene));
    let ctl = Control::new(600, 1);
    for _ in 0..600 {
        if a2.step(&ctl).converged {
            break;
        }
    }

    let l0 = Problem::new(
        &system,
        &counts,
        &flux,
        Fidelity::L0,
        golden.exposure_s,
        golden.scene.forward_params(),
    );
    let mut a1 = TransmissionEm::new(&l0, A1Config::from_scene(&golden.scene));
    for _ in 0..600 {
        if a1.step(&ctl).converged {
            break;
        }
    }

    let a2_fit = l1.nll(a2.estimate());
    let a1_fit = l1.nll(a1.estimate());
    println!("L1 fit: A2 {a2_fit:.6e}, A1 (L0 model) {a1_fit:.6e}");
    assert!(
        a2_fit < a1_fit,
        "the L1 model should fit L1 data better: A2 {a2_fit:.6e} vs A1 {a1_fit:.6e}"
    );
}

#[test]
fn cancellation_leaves_the_estimate_where_it_was() {
    let fixture = small();
    let problem = fixture.problem();
    let mut solver = PoissonMap::new(&problem, fixture.cfg.clone());
    let ctl = Control::new(50, 1);
    solver.step(&ctl);
    let before = solver.estimate().to_vec();
    let objective = solver.objective();

    ctl.request_stop();
    let report = solver.step(&ctl);
    assert_eq!(solver.estimate(), before.as_slice());
    assert_eq!(report.objective, Some(objective));
}
