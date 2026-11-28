//! Calibration campaigns: SBC (A11) and the parametric bootstrap (A11).
//!
//! Both commands exist so a calibration claim can be **re-run from a seed**
//! rather than believed from a sentence. They share the scene loader, the
//! solver dispatch and the report format; they differ in what they compare:
//!
//! | Command | Question |
//! |---|---|
//! | `umbra sbc` | Do the engine's posterior intervals contain the truth at the rate they claim? |
//! | `umbra bootstrap` | What is the empirical σ map at a fit, and does it cover? |
//!
//! # What the SBC engine is
//!
//! `--engine a1|a2` selects the *fit*; the posterior around that fit is the
//! Laplace approximation of [`umbra_stats::laplace`] — per-voxel marginal
//! Gaussians at the fitted field. That is deliberate: the result cards report
//! exactly those bars, so what SBC has to check is the pipeline the page
//! actually runs, not a posterior the page never computes. Where the Laplace
//! marginals are wrong, the rank histogram says so, and that is a finding about
//! the product rather than about the harness.
//!
//! A7 has no engine yet (#18), so `--engine a7` is refused with the reason
//! instead of quietly running A2 under another name.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use umbra_core::rng::Philox;
use umbra_forward::expected;
use umbra_forward::scene::SceneSpec;
use umbra_forward::{build_system_2d, System};
use umbra_physics::{generate_default, Fidelity, FluxTable};
use umbra_recon::a1::{A1Config, TransmissionEm};
use umbra_recon::a2::{A2Config, PoissonMap};
use umbra_recon::{Control, Problem, Solver, Start};
use umbra_stats::bootstrap::{bootstrap_sigma, coverage, wilson};
use umbra_stats::laplace::{poisson_laplace, LaplaceConfig, PoissonProblem, SigmaMap, Uncertainty};
use umbra_stats::sbc::{run as run_sbc, Engine, SbcConfig, SbcReport};

/// The shared default scene.
const DEFAULT_SCENE: &str = "web/src/scenes/default.json";

/// Which fit the campaign drives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    /// A1, transmission EM. L0 only, as the solver itself insists.
    A1,
    /// A2, Poisson MAP with smoothed TV.
    A2,
}

impl Kind {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "a1" => Some(Kind::A1),
            "a2" => Some(Kind::A2),
            _ => None,
        }
    }

    fn id(self) -> &'static str {
        match self {
            Kind::A1 => "a1",
            Kind::A2 => "a2",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Kind::A1 => "transmission EM, L0 only",
            Kind::A2 => "Poisson MAP with Huber TV",
        }
    }
}

/// `--name VALUE` as text.
fn text(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

/// `--name VALUE` as a number.
fn number(args: &[String], name: &str) -> Option<f64> {
    text(args, name).and_then(|value| value.trim().parse::<f64>().ok())
}

/// Loads the scene the campaign is about, with a named error.
fn load(scene: &Path) -> Result<SceneSpec, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(scene)?;
    Ok(SceneSpec::from_json_str(&text)?)
}

/// Which engine, or why not.
fn engine(args: &[String]) -> Result<Kind, Box<dyn std::error::Error>> {
    let name = text(args, "--engine").unwrap_or_else(|| "a2".to_string());
    if name == "a7" {
        return Err(
            "engine `a7` does not exist yet: the void model is #18, and this harness will \
             register it when it lands rather than approximate it with A2"
                .into(),
        );
    }
    Kind::parse(&name).ok_or_else(|| {
        format!("unknown engine `{name}`: expected a1, a2 or a7 (a7 is not implemented yet)").into()
    })
}

/// Which curvature the Laplace bars are built from.
///
/// The product reports the majoriser, and the campaign measures it against the
/// exact curvature at the same estimate. Keeping both behind a flag is what
/// makes the difference a measurement rather than a swap: the two maps share
/// one walk over the grid and differ only in which edges carry `β/ε`, so a
/// disagreement in the report is about the prior and nothing else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Prior {
    /// `β/ε` on every edge — what the result card shows today.
    Majoriser,
    /// `β/ε` only where the fitted difference is inside `ε`.
    Exact,
}

impl Prior {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "majoriser" => Some(Prior::Majoriser),
            "exact" => Some(Prior::Exact),
            _ => None,
        }
    }

    fn id(self) -> &'static str {
        match self {
            Prior::Majoriser => "majoriser",
            Prior::Exact => "exact",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Prior::Majoriser => "β/ε on every edge (the card's bars)",
            Prior::Exact => "β/ε inside the Huber band, 0 on a jump",
        }
    }
}

/// Which prior curvature, or why not.
fn prior(args: &[String]) -> Result<Prior, Box<dyn std::error::Error>> {
    let name = text(args, "--prior").unwrap_or_else(|| "majoriser".to_string());
    Prior::parse(&name)
        .ok_or_else(|| format!("unknown prior `{name}`: expected majoriser or exact").into())
}

/// One fit of `counts` with the selected solver.
fn fit(kind: Kind, problem: &Problem<'_>, rho_max: f32, max_iters: u32) -> Vec<f32> {
    match kind {
        Kind::A1 => {
            let cfg = A1Config {
                rho_max,
                ..A1Config::default()
            };
            let mut solver = TransmissionEm::init(problem, &cfg, &Start::default());
            let ctl = Control::new(max_iters, max_iters);
            let mut report = solver.step(&ctl);
            while !report.converged && report.iteration < max_iters {
                report = solver.step(&ctl);
            }
            solver.estimate().to_vec()
        }
        Kind::A2 => {
            let cfg = A2Config {
                rho_max,
                ..A2Config::default()
            };
            let mut solver = PoissonMap::init(problem, &cfg, &Start::default());
            let ctl = Control::new(max_iters, max_iters);
            let mut report = solver.step(&ctl);
            while !report.converged && report.iteration < max_iters {
                report = solver.step(&ctl);
            }
            solver.estimate().to_vec()
        }
    }
}

/// Voxels at least one ray can see. The rest cannot be ranked: their posterior
/// is the prior, whatever the prior is, so calling a rank histogram there a
/// calibration check would be testing the harness against itself.
fn observable(system: &System) -> Vec<usize> {
    let mut seen = vec![false; system.voxel_count()];
    for (row, ray) in system.rays.iter().enumerate() {
        if ray.k_scale <= 0.0 {
            continue;
        }
        for (voxel, length) in system.l.row(row) {
            if length > 0.0 {
                seen[voxel] = true;
            }
        }
    }
    seen.iter()
        .enumerate()
        .filter_map(|(index, value)| value.then_some(index))
        .collect()
}

/// A standard normal draw, Box–Muller.
fn normal(rng: &mut Philox) -> f64 {
    let u1 = rng.uniform().max(1e-300);
    let u2 = rng.uniform();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The pipeline under test, as an SBC engine: prior over the box, Poisson
/// simulation, and the fit-plus-Laplace posterior the card reports.
struct Pipeline {
    system: System,
    flux: FluxTable,
    fidelity: Fidelity,
    exposure_s: f32,
    params: umbra_forward::expected::ForwardParams,
    kind: Kind,
    prior: Prior,
    posterior: usize,
    inflate: f32,
    rho_max: f32,
    mask: Vec<usize>,
    fits: u64,
}

impl Pipeline {
    /// `mask`-length parameters expanded to a full density field.
    ///
    /// Voxels outside the mask cannot affect a ray, so zero is not a choice the
    /// data can distinguish: it is a fixed value for entries the posterior never
    /// reports.
    fn expand(&self, theta: &[f32]) -> Vec<f32> {
        let mut rho = vec![0.0f32; self.system.voxel_count()];
        for (slot, voxel) in self.mask.iter().enumerate() {
            rho[*voxel] = theta[slot];
        }
        rho
    }

    fn fit_at(&mut self, counts: &[f32]) -> Vec<f32> {
        self.fits += 1;
        let problem = Problem::new(
            &self.system,
            counts,
            &self.flux,
            self.fidelity,
            self.exposure_s,
            self.params,
        );
        fit(self.kind, &problem, self.rho_max, 400)
    }

    /// The error bars the result card would show at `rho`, with the prior the
    /// estimate was produced under.
    ///
    /// The TV curvature is not optional here: at 32 rays on a 1024-voxel grid
    /// the data-only Fisher matrix is singular, the conjugate-gradient solve
    /// diverges to variances of `10²⁸`, and every rank lands at 0.5 — a
    /// calibration report made of one number. The card builds the same prior for
    /// the same reason, so the campaign must too.
    fn sigma_map(&self, rho: &[f32]) -> SigmaMap {
        let counts = vec![0.0f32; self.system.ray_count()];
        let problem = Problem::new(
            &self.system,
            &counts,
            &self.flux,
            self.fidelity,
            self.exposure_s,
            self.params,
        );
        let prior = match self.kind {
            Kind::A2 => {
                let cfg = A2Config::default();
                Some(match self.prior {
                    Prior::Majoriser => {
                        umbra_recon::tv_curvature(&problem, rho, cfg.beta, cfg.huber_eps)
                    }
                    Prior::Exact => {
                        umbra_recon::tv_curvature_at(&problem, rho, cfg.beta, cfg.huber_eps)
                    }
                })
            }
            // No TV term, so no prior curvature: A1's card is the data-only
            // Fisher map, which on a limited-angle scene is singular and will
            // report itself unresolved rather than produce bars.
            Kind::A1 => None,
        };
        let observation = PoissonProblem {
            system: &self.system,
            flux: &self.flux,
            fidelity: self.fidelity,
            exposure_s: self.exposure_s,
            params: self.params,
        };
        poisson_laplace(&observation, rho, prior.as_ref(), &LaplaceConfig::default())
    }
}

impl Engine for Pipeline {
    fn parameter_count(&self) -> usize {
        self.mask.len()
    }

    fn prior(&mut self, rng: &mut Philox) -> Vec<f32> {
        // Uniform on the box every solver already projects onto. A smoother
        // prior would be an assumption the pipeline does not make, and SBC is
        // only meaningful against the prior the engine actually sees.
        (0..self.mask.len())
            .map(|_| (rng.uniform() * f64::from(self.rho_max)) as f32)
            .collect()
    }

    fn simulate(&self, theta: &[f32], rng: &mut Philox) -> Vec<f32> {
        let lambda = expected(
            &self.system,
            &self.expand(theta),
            &self.flux,
            self.fidelity,
            self.exposure_s,
            &self.params,
        )
        .lambda;
        lambda
            .iter()
            .map(|value| rng.poisson(f64::from(*value).max(0.0)) as f32)
            .collect()
    }

    fn posterior(&mut self, _theta: &[f32], data: &[f32], rng: &mut Philox) -> Vec<Vec<f32>> {
        let rho = self.fit_at(data);
        let map = self.sigma_map(&rho);
        if !map.is_converged() {
            // The card's own rule: bars from a solve that missed its tolerance
            // are not error bars. Contributing them would put a plausible rank
            // on a number that has not converged, so the draw is declined and
            // counted instead.
            return Vec::new();
        }
        let mut draws = Vec::with_capacity(self.posterior);
        for _ in 0..self.posterior {
            let mut sample = Vec::with_capacity(self.mask.len());
            for voxel in &self.mask {
                let centre = f64::from(rho[*voxel]);
                let value = match map.at(*voxel) {
                    Uncertainty::Sigma(sigma) => {
                        centre + self.inflate as f64 * f64::from(sigma) * normal(rng)
                    }
                    // A voxel the geometry cannot see has no interval to sample
                    // from, but the mask already excludes it.
                    Uncertainty::Unconstrained => centre,
                };
                sample.push(value as f32);
            }
            draws.push(sample);
        }
        draws
    }

    fn name(&self) -> &str {
        self.kind.id()
    }
}

/// Builds the pipeline for a scene.
fn pipeline(
    spec: &SceneSpec,
    kind: Kind,
    prior: Prior,
    posterior: usize,
    inflate: f32,
) -> Result<Pipeline, Box<dyn std::error::Error>> {
    if kind == Kind::A1 && spec.physics.level != Fidelity::L0 {
        return Err(format!(
            "engine a1 is an L0 method and this scene is {}; use --engine a2 or an L0 scene",
            "not L0"
        )
        .into());
    }
    let system = build_system_2d(&spec.to_scene_2d());
    let mask = observable(&system);
    if mask.is_empty() {
        return Err("no voxel is observable: the scene has no live ray".into());
    }
    let rho_max = A1Config::from_scene(spec).rho_max;
    Ok(Pipeline {
        system,
        flux: generate_default(),
        fidelity: spec.physics.level,
        exposure_s: spec.exposure_s,
        params: spec.forward_params(),
        kind,
        prior,
        posterior,
        inflate,
        rho_max,
        mask,
        fits: 0,
    })
}

/// `umbra sbc`.
pub fn sbc(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let scene_path = text(args, "--scene")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE));
    let spec = load(&scene_path)?;
    let kind = engine(args)?;
    let prior = prior(args)?;
    let config = SbcConfig {
        draws: number(args, "--draws").unwrap_or(500.0) as u32,
        posterior: number(args, "--posterior").unwrap_or(100.0) as u32,
        seed: number(args, "--seed").unwrap_or(7.0) as u64,
        bins: 10,
    };
    let inflate = number(args, "--inflate").unwrap_or(1.0) as f32;
    let report_path = text(args, "--report")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/sbc/report.md"));

    let mut engine = pipeline(&spec, kind, prior, config.posterior as usize, inflate)?;
    let observable = engine.mask.len();
    let start = Instant::now();
    let report = run_sbc(&mut engine, config);
    let elapsed = start.elapsed().as_secs_f64();
    let fits = engine.fits;

    let document = sbc_document(
        &spec,
        &scene_path,
        &report_path,
        kind,
        prior,
        inflate,
        &Run {
            observable,
            fits,
            elapsed,
        },
        &report,
    );
    write(&report_path, &document)?;
    println!(
        "sbc: engine {} draws {} pooled D {:.4} p {:.3e} min p {:.3e} fail {} -> {}",
        kind.id(),
        report.config.draws,
        report.pooled_ks,
        report.pooled_p,
        report.min_p,
        report.failures,
        report_path.display()
    );
    if !report.passed() {
        // A calibration campaign that fails is data, not an infrastructure
        // fault: print the verdict and exit 0 so a recorded run can carry it.
        println!("verdict: FAIL (the ranks are not uniform at p = 0.01)");
    }
    Ok(())
}

/// What the run cost and how much of it was usable.
struct Run {
    observable: usize,
    fits: u64,
    elapsed: f64,
}

#[allow(clippy::too_many_arguments)]
fn sbc_document(
    spec: &SceneSpec,
    scene: &Path,
    report_path: &Path,
    kind: Kind,
    prior: Prior,
    inflate: f32,
    run: &Run,
    report: &SbcReport,
) -> String {
    let Run {
        observable,
        fits,
        elapsed,
    } = *run;
    let mut worst: Vec<_> = report.parameters.iter().collect();
    worst.sort_by(|a, b| a.p.partial_cmp(&b.p).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = String::new();
    out.push_str(&format!("# SBC — {} on `{}`\n\n", kind.id(), spec.id));
    out.push_str(
        "Simulation-based calibration (Talts et al. 2018): draw the truth from the prior, \
         simulate counts, run the engine, and rank the truth among its posterior draws. \
         A calibrated engine gives uniform ranks.\n\n",
    );
    out.push_str("| | |\n|---|---|\n");
    out.push_str(&format!("| Scene | `{}` ({}) |\n", spec.id, spec.label));
    out.push_str(&format!(
        "| Grid | {} × {} at {} m |\n",
        spec.grid.n[0], spec.grid.n[1], spec.grid.h_m
    ));
    out.push_str(&format!(
        "| Engine | {} ({}) |\n",
        kind.id(),
        kind.description()
    ));
    out.push_str(&format!(
        "| Prior curvature | {} ({}) |\n",
        prior.id(),
        prior.description()
    ));
    out.push_str(&format!(
        "| Posterior | Laplace marginals at the fit, {} draws |\n",
        report.config.posterior
    ));
    out.push_str(&format!("| Draws | {} |\n", report.config.draws));
    out.push_str(&format!("| Observable voxels | {observable} |\n"));
    out.push_str(&format!("| Seed | `{:#x}` |\n", report.config.seed));
    out.push_str(&format!("| Inflation | {}× |\n", inflate));
    out.push_str(&format!("| Fits | {fits} |\n"));
    out.push_str(&format!(
        "| Runtime | {:.1} s ({:.1} ms/draw) |\n",
        elapsed, report.ms_per_draw
    ));
    out.push_str(&format!(
        "| Declined draws | {} (the Laplace solve did not converge) |\n",
        report.missing
    ));
    out.push_str(&format!(
        "| Pooled KS | D = {:.4}, p = {:.3e} |\n",
        report.pooled_ks, report.pooled_p
    ));
    out.push_str(&format!(
        "| Per-parameter p | median {:.3}, min {:.3}, {} below 0.01 |\n\n",
        report.median_p, report.min_p, report.failures
    ));

    out.push_str("## Rank histograms\n\n");
    out.push_str(
        "Ten bins over `[0, 1)`. Uniform requires the same count in each; the two \
         end bins are where an over-confident engine shows up, and the middle pair \
         is where an under-confident one does.\n\n",
    );
    out.push_str("| Parameter | bins | KS D | p |\n|---|---|---|---|\n");
    for parameter in worst.iter().take(10) {
        out.push_str(&format!(
            "| {} | {:?} | {:.4} | {:.3e} |\n",
            parameter.index, parameter.histogram, parameter.ks, parameter.p
        ));
    }
    out.push_str(&format!(
        "\nThe ten lowest p-values of {}. The gate is the pooled p, not these: a \
         thousand parameters put one below 0.01 about every ten runs even when the \
         engine is correct.\n\n",
        report.parameters.len()
    ));

    out.push_str("## What a decision at 10⁴ draws would cost\n\n");
    let projected = 10_000.0 * report.ms_per_draw / 1000.0;
    out.push_str(&format!(
        "At the measured {:.1} ms/draw, 10⁴ draws take {:.0} s — {:.1} h. That is a CI \
         job rather than a laptop run, which is why this report records the size it \
         actually ran, and why the scheduled campaign runs the spec's size as four \
         shards of 2500 draws — one job each, so no shard outlives a runner — behind \
         `docs/calibration/sbc-recorded.txt`.\n\n",
        report.ms_per_draw,
        projected,
        projected / 3600.0
    ));

    out.push_str("## Verdict\n\n");
    out.push_str(&format!(
        "Pooled KS p = {:.3e} against the 0.01 gate: **{}**.\n\n",
        report.pooled_p,
        if report.passed() { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!("{}\n", report.verdict_line()));
    out.push_str(&format!(
        "\nReproduced by `cargo run --release -p umbra-cli -- sbc --scene {} --engine {} \
         --prior {} --draws {} --posterior {} --seed {} --inflate {} --report {}`.\n",
        scene.display(),
        kind.id(),
        prior.id(),
        report.config.draws,
        report.config.posterior,
        report.config.seed,
        inflate,
        report_path.display()
    ));
    out
}

/// `umbra bootstrap`.
pub fn bootstrap(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let scene_path = text(args, "--scene")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE));
    let spec = load(&scene_path)?;
    let kind = engine(args)?;
    let prior = prior(args)?;
    let datasets = number(args, "--datasets").unwrap_or(30.0) as u32;
    let replicates = number(args, "--replicates").unwrap_or(40.0) as u32;
    let seed = number(args, "--seed").unwrap_or(7.0) as u64;
    let report_path = text(args, "--report")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/bootstrap/report.md"));

    let mut engine = pipeline(&spec, kind, prior, 0, 1.0)?;
    let truth = spec.density();
    let fisher_map = engine.sigma_map(&truth);
    let sigma_fisher = &fisher_map.sigma;
    let start = Instant::now();

    // Coverage is measured over independent datasets at the same truth, and
    // inside each one over the voxels: one dataset's voxels are correlated by
    // the operator, so the Wilson interval is optimistic in the voxel direction
    // and the table reports both counts rather than only the rate.
    let mut inside = 0u32;
    let mut centred = 0u32;
    let mut total = 0u32;
    let mut unusable = 0u32;
    let mut bias_total = 0.0f64;
    let mut ratios: Vec<f32> = Vec::new();
    let mut ms_per_replicate = 0.0f64;
    let mut replicate_count = 0u32;

    for dataset in 0..datasets {
        let counts = engine.simulate(
            &truth,
            &mut Philox::new(seed, 0xDA_7A_00 + u64::from(dataset)),
        );
        let estimate = engine.fit_at(&counts);
        let problem = PoissonProblem {
            system: &engine.system,
            flux: &engine.flux,
            fidelity: engine.fidelity,
            exposure_s: engine.exposure_s,
            params: engine.params,
        };
        let rho_max = engine.rho_max;
        let map = bootstrap_sigma(
            &problem,
            &estimate,
            replicates,
            seed ^ (u64::from(dataset) << 32),
            |counts: &[f32]| {
                let problem = Problem::new(
                    &engine.system,
                    counts,
                    &engine.flux,
                    engine.fidelity,
                    engine.exposure_s,
                    engine.params,
                );
                fit(kind, &problem, rho_max, 400)
            },
        );
        ms_per_replicate += map.ms_per_replicate;
        replicate_count += 1;
        let measured = coverage(&estimate, &map.sigma, &truth);
        inside += measured.inside;
        total += measured.total;
        unusable += measured.unusable;
        // The same intervals, centred on the mean refit instead of the estimate.
        // Coverage against the truth mixes two failures — bars that are too
        // narrow and an estimate that sits off the truth — and splitting them is
        // what says which one the pipeline has.
        let about_mean = coverage(&estimate, &map.sigma, &map.mean);
        centred += about_mean.inside;
        for bias in &map.bias {
            bias_total += f64::from(bias.abs());
        }
        for (voxel, sigma) in map.sigma.iter().enumerate() {
            if *sigma > 0.0 {
                if let Uncertainty::Sigma(fisher) = sigma_fisher[voxel] {
                    ratios.push(*sigma / fisher);
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let measured = coverage_rate(inside, total);
    let document = bootstrap_document(
        &spec,
        &scene_path,
        &report_path,
        kind,
        datasets,
        replicates,
        seed,
        elapsed,
        ms_per_replicate / f64::from(replicate_count.max(1)),
        &measured,
        prior,
        if total == 0 {
            0.0
        } else {
            f64::from(centred) / f64::from(total)
        },
        if total == 0 {
            0.0
        } else {
            bias_total / f64::from(total)
        },
        unusable,
        &ratios,
        &fisher_map,
    );
    write(&report_path, &document)?;
    println!(
        "bootstrap: engine {} datasets {} replicates {} coverage {:.3} [{:.3}, {:.3}] -> {}",
        kind.id(),
        datasets,
        replicates,
        measured.0,
        measured.1,
        measured.2,
        report_path.display()
    );
    Ok(())
}

/// Rate, low and high, kept together so the report cannot quote one without the
/// others.
fn coverage_rate(inside: u32, total: u32) -> (f64, f64, f64) {
    let (low, high) = wilson(inside, total);
    let rate = if total == 0 {
        0.0
    } else {
        f64::from(inside) / f64::from(total)
    };
    (rate, low, high)
}

#[allow(clippy::too_many_arguments)]
fn bootstrap_document(
    spec: &SceneSpec,
    scene: &Path,
    report_path: &Path,
    kind: Kind,
    datasets: u32,
    replicates: u32,
    seed: u64,
    elapsed: f64,
    ms_per_replicate: f64,
    measured: &(f64, f64, f64),
    prior: Prior,
    centred_rate: f64,
    mean_bias: f64,
    unusable: u32,
    ratios: &[f32],
    fisher: &SigmaMap,
) -> String {
    let constrained = fisher.sigma.iter().filter(|s| s.is_constrained()).count();
    let mut sorted = ratios.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(f32::NAN);

    let mut out = String::new();
    out.push_str(&format!(
        "# Parametric bootstrap — {} on `{}`\n\n",
        kind.id(),
        spec.id
    ));
    out.push_str(
        "Counts are re-drawn from the fitted `λ̂` and the same solver refits each replicate. \
         The spread of those refits is the empirical σ map; the Laplace map is the local \
         Gaussian answer the cards already show. Where the two disagree is where the \
         approximation is not enough.\n\n",
    );
    out.push_str("| | |\n|---|---|\n");
    out.push_str(&format!("| Scene | `{}` ({}) |\n", spec.id, spec.label));
    out.push_str(&format!(
        "| Engine | {} ({}) |\n",
        kind.id(),
        kind.description()
    ));
    out.push_str(&format!(
        "| Prior curvature | {} ({}) |\n",
        prior.id(),
        prior.description()
    ));
    out.push_str(&format!("| Datasets | {datasets} |\n"));
    out.push_str(&format!("| Replicates each | {replicates} |\n"));
    out.push_str(&format!("| Seed | `{:#x}` |\n", seed));
    out.push_str(&format!(
        "| Cost | {:.1} ms/replicate, {:.1} s total |\n",
        ms_per_replicate, elapsed
    ));
    out.push_str(&format!(
        "| Observable voxels | {} of {} |\n",
        constrained,
        fisher.len()
    ));
    out.push_str(&format!(
        "| Fisher solve | {} unresolved, worst residual {:.2e} |\n\n",
        fisher.unresolved, fisher.worst_residual
    ));

    out.push_str("## Coverage at the stated rate\n\n");
    out.push_str(
        "Nominal 68.27%: the fraction of voxels with `|ρ̂ − ρ_true| ≤ σ̂`. Voxels inside \
         one dataset are correlated by the operator, so the Wilson interval is a \
         voxel-count interval and not an independent-sample one — the honest summary is \
         the rate, with the interval read as an indication of how far the count could \
         sit from it.\n\n",
    );
    out.push_str("| | |\n|---|---|\n");
    out.push_str(&format!("| Observed | {:.3} |\n", measured.0));
    out.push_str(&format!(
        "| Wilson 95% | [{:.3}, {:.3}] |\n",
        measured.1, measured.2
    ));
    out.push_str(&format!(
        "| Contains 0.6827 | {} |\n",
        if measured.1 <= 0.6827 && 0.6827 <= measured.2 {
            "yes"
        } else {
            "no"
        }
    ));
    out.push_str(&format!("| Voxels with no usable σ | {unusable} |\n\n"));

    out.push_str("## The same intervals, centred on the refit mean\n\n");
    out.push_str(
        "Coverage against the truth measures two things at once: whether the bars are \
         the right width, and whether the estimate sits on the truth at all. Centring \
         the interval on the mean refit removes the second, so a rate near nominal \
         here with a low rate above says the bars are fine and the *estimate* is \
         biased; a low rate in both says the width itself is wrong.\n\n",
    );
    out.push_str("| | |\n|---|---|\n");
    out.push_str(&format!("| About the refit mean | {:.3} |\n", centred_rate));
    out.push_str(&format!(
        "| Mean absolute bias | {mean_bias:.4} g cm⁻³ |\n\n"
    ));

    out.push_str("## Bootstrap σ against the Fisher bars\n\n");
    out.push_str(&format!(
        "Median ratio of bootstrap σ to Fisher σ over every covered voxel: **{:.2}×**. \
         A ratio below 1 means the Laplace bars are wider than the refits actually move, \
         which is what a prior does where the data are weak; above 1 means they are too \
         narrow.\n\n",
        median
    ));

    out.push_str("## What a decision would need\n\n");
    out.push_str(&format!(
        "A ±2% decision on the mean spread needs {} replicates per dataset \
         (`1/0.02²`). At {:.1} ms each that is {:.1} min per dataset, and this \
         report's {replicates} replicates give a ±{:.1}% interval on the spread.\n\n",
        umbra_stats::bootstrap::BootstrapSigma::replicates_to_decide(0.02).unwrap_or(0),
        ms_per_replicate,
        ms_per_replicate * 2500.0 / 60_000.0,
        100.0 / (f64::from(replicates).max(1.0)).sqrt()
    ));

    out.push_str("## Verdict\n\n");
    out.push_str(&format!(
        "Coverage {:.3} with Wilson [{:.3}, {:.3}]; nominal 0.6827 is {} the interval.\n\n",
        measured.0,
        measured.1,
        measured.2,
        if measured.1 <= 0.6827 && 0.6827 <= measured.2 {
            "inside"
        } else {
            "outside"
        }
    ));
    out.push_str(&format!(
        "<!-- bootstrap-verdict engine={} datasets={} replicates={} coverage={:.4} low={:.4} high={:.4} contains_nominal={} -->\n",
        kind.id(),
        datasets,
        replicates,
        measured.0,
        measured.1,
        measured.2,
        measured.1 <= 0.6827 && 0.6827 <= measured.2
    ));
    out.push_str(&format!(
        "\nReproduced by `cargo run --release -p umbra-cli -- bootstrap --scene {} \
         --engine {} --prior {} --datasets {datasets} --replicates {replicates} --seed {} \
         --report {}`.\n",
        scene.display(),
        kind.id(),
        prior.id(),
        seed,
        report_path.display()
    ));
    out
}

/// Writes a report, creating its directory.
fn write(path: &Path, document: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, document)?;
    Ok(())
}
