//! Runtime benchmark for the reconstruction portfolio on the shared default scene.
//!
//! One dataset and one metric, so A1–A4's numbers can sit in the same table
//! instead of four ad-hoc ones: every solver starts from the bulk density and is
//! timed to a fixed **data fit** — the Poisson NLL of its estimate, computed
//! here. The fit is measured from outside the solver because a regularised
//! solver's own objective contains its prior: A2's `objective()` is lower than
//! A1's at the same fit, so comparing those values would reward the size of the
//! penalty rather than the quality of the reconstruction.
//!
//! Adding a solver is one line in [`portfolio`]. Nothing is warm-started, so a
//! method cannot be flattered by starting closer to the answer than its rival.
//!
//! Deliberately dependency-free: `criterion` 0.5 and 0.4 both pull in
//! `windows-sys` with raw-dylib imports, which needs `dlltool` to link on an
//! `x86_64-pc-windows-gnu` host, so the benchmark would not build on the machine
//! its author uses. `std::time::Instant` over a warm run costs nothing and keeps
//! `umbra-recon`'s dependency list empty.
//!
//! Run with `cargo bench -p umbra-recon --bench portfolio`.

use std::path::Path;
use std::time::{Duration, Instant};

use umbra_forward::expected::ForwardParams;
use umbra_forward::scene::SceneSpec;
use umbra_forward::{build_system_2d, expected};
use umbra_physics::{generate_default, Fidelity, FluxTable};
use umbra_recon::{
    A1Config, A2Config, A3Config, Control, OpacityPdhg, PoissonMap, Problem, Solver, StepReport,
    TransmissionEm,
};

/// The data-fit level every solver is timed to, in NLL.
///
/// Reached well before A1 converges (about `-3.5379e8`), so the measurement
/// covers the fast part of the descent — the part an interactive slider waits
/// for — rather than the flat tail where the portfolio's methods differ by a few
/// parts in `e-5`.
const NLL_TARGET: f64 = -3.5e8;

/// Steps allowed before a run to the target is abandoned.
const STEP_BUDGET: u32 = 2_000;

/// The shared scene at L0, the counts its own model predicts, and the model.
struct Data {
    system: umbra_forward::System,
    counts: Vec<f32>,
    flux: FluxTable,
    params: ForwardParams,
    exposure_s: f32,
}

impl Data {
    fn problem(&self) -> Problem<'_> {
        Problem::new(
            &self.system,
            &self.counts,
            &self.flux,
            Fidelity::L0,
            self.exposure_s,
            self.params,
        )
    }

    /// Poisson NLL of an estimate under the shared scene's L0 model.
    fn nll(&self, rho: &[f32]) -> f64 {
        self.problem().nll(rho)
    }
}

fn shared_scene() -> SceneSpec {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/src/scenes/default.json");
    let text = std::fs::read_to_string(&path).expect("the committed default scene is readable");
    let mut scene = SceneSpec::from_json_str(&text).expect("the committed default scene loads");
    scene.physics.level = Fidelity::L0;
    scene
}

fn dataset(scene: &SceneSpec) -> Data {
    let system = build_system_2d(&scene.to_scene_2d());
    let flux = generate_default();
    let params = scene.forward_params();
    let counts = expected(
        &system,
        &scene.density(),
        &flux,
        Fidelity::L0,
        scene.exposure_s,
        &params,
    )
    .lambda;
    Data {
        system,
        counts,
        flux,
        params,
        exposure_s: scene.exposure_s,
    }
}

/// A solver with its config and type erased, so one loop can hold the portfolio.
///
/// [`Solver`] itself is not object-safe — `init` is generic over the config and
/// the starting point — but stepping and reading the estimate are, which is all
/// a benchmark needs.
trait Stepper {
    fn step(&mut self, ctl: &Control) -> StepReport;
    fn estimate(&self) -> &[f32];
}

impl<S: Solver> Stepper for S {
    fn step(&mut self, ctl: &Control) -> StepReport {
        <S as Solver>::step(self, ctl)
    }

    fn estimate(&self) -> &[f32] {
        <S as Solver>::estimate(self)
    }
}

/// One portfolio entry: an id and a way to build it from the shared data.
type Factory = Box<dyn Fn(&Data) -> Box<dyn Stepper>>;

struct Entry {
    id: &'static str,
    make: Factory,
}

/// The portfolio, in the order the numbers should be read.
fn portfolio(scene: &SceneSpec) -> Vec<Entry> {
    let a1 = A1Config {
        subsets: 8,
        ..A1Config::from_scene(scene)
    };
    let a2 = A2Config::from_scene(scene);
    let a3 = A3Config::from_scene(scene);
    vec![
        Entry {
            id: "a1-transmission-em",
            make: Box::new(move |data| Box::new(TransmissionEm::new(&data.problem(), a1.clone()))),
        },
        Entry {
            id: "a2-poisson-map",
            make: Box::new(move |data| Box::new(PoissonMap::new(&data.problem(), a2.clone()))),
        },
        Entry {
            id: "a3-opacity-pdhg",
            make: Box::new(move |data| Box::new(OpacityPdhg::new(&data.problem(), a3.clone()))),
        },
    ]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1.0e3
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Cost of one step, with construction excluded: a caller builds a solver once
/// and steps it many times, so the step is what a slider move actually pays.
fn step_cost(data: &Data, entry: &Entry, runs: u32) -> Duration {
    let mut samples = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let mut solver = (entry.make)(data);
        let ctl = Control::new(1, 1);
        let started = Instant::now();
        solver.step(&ctl);
        samples.push(started.elapsed());
    }
    median(samples)
}

/// What one run to the shared fit target cost.
struct Reached {
    steps: u32,
    fit: f64,
    elapsed: Duration,
}

fn to_target(data: &Data, entry: &Entry) -> Reached {
    let mut solver = (entry.make)(data);
    let ctl = Control::new(STEP_BUDGET, 1);
    let started = Instant::now();
    let mut steps = 0;
    loop {
        let report = solver.step(&ctl);
        steps += 1;
        let fit = data.nll(solver.estimate());
        if fit <= NLL_TARGET || report.converged || steps >= STEP_BUDGET {
            return Reached {
                steps,
                fit,
                elapsed: started.elapsed(),
            };
        }
    }
}

fn main() {
    let scene = shared_scene();
    let data = dataset(&scene);
    let entries = portfolio(&scene);
    println!(
        "portfolio on the shared scene at L0: {} rays, {} voxels",
        data.system.ray_count(),
        data.system.voxel_count()
    );
    println!("fit target NLL {NLL_TARGET:.3e}; every solver starts from the bulk density");

    for entry in &entries {
        // Warm-up: the first runs pay for page faults and the allocator, which
        // has nothing to do with the algorithm.
        for _ in 0..3 {
            let _ = to_target(&data, entry);
        }

        println!(
            "{:<22} one step   median {:>8.3} ms",
            entry.id,
            ms(step_cost(&data, entry, 200))
        );

        let mut samples = Vec::new();
        let mut last = None;
        for _ in 0..10 {
            let reached = to_target(&data, entry);
            samples.push(reached.elapsed);
            last = Some(reached);
        }
        let reached = last.expect("ten runs were requested");
        println!(
            "{:<22} to {:<9.3e} median {:>8.3} ms   ({} steps, fit {:.6e})",
            entry.id,
            NLL_TARGET,
            ms(median(samples)),
            reached.steps,
            reached.fit
        );
    }
}
