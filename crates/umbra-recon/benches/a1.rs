//! Runtime benchmark for A1 on the shared default scene.
//!
//! Deliberately dependency-free: criterion 0.5 and 0.4 both pull in
//! `windows-sys` with raw-dylib imports, which needs `dlltool` to link on an
//! `x86_64-pc-windows-gnu` host, so the benchmark would fail to build on the
//! machine its author uses. `std::time::Instant` over a warm run costs nothing
//! and keeps `umbra-recon`'s dependency list empty.
//!
//! The dataset is the committed scene at L0 — A1 is L0-only, and the golden
//! file holds L1 counts — evaluated here rather than read from JSON, so the
//! numbers measure the solver and not the loader.
//!
//! Run with `cargo bench -p umbra-recon --bench a1`. Numbers are printed as
//! `name  min / median / mean` in milliseconds; the reconstruction figure is
//! the one to quote.

use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use umbra_forward::scene::SceneSpec;
use umbra_forward::{build_system_2d, expected, System};
use umbra_physics::{generate_default, Fidelity};
use umbra_recon::{A1Config, Control, Solver, TransmissionEm};

/// The committed scene at L0, with counts that are its own prediction.
fn dataset() -> (System, Vec<f32>, A1Config) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/src/scenes/default.json");
    let text = std::fs::read_to_string(&path).expect("the committed default scene is readable");
    let mut scene = SceneSpec::from_json_str(&text).expect("the committed default scene loads");
    scene.physics.level = Fidelity::L0;

    let system = build_system_2d(&scene.to_scene_2d());
    let model = expected(
        &system,
        &scene.density(),
        &generate_default(),
        Fidelity::L0,
        scene.exposure_s,
        &scene.forward_params(),
    );
    let cfg = A1Config {
        subsets: 8,
        ..A1Config::from_scene(&scene)
    };
    (system, model.lambda, cfg)
}

fn one_step(system: &System, counts: &[f32], cfg: &A1Config) {
    let mut solver = TransmissionEm::new(system, counts, cfg.clone());
    let ctl = Control::new(1, 1);
    solver.step(&ctl);
    black_box(solver.objective());
}

fn reconstruct(system: &System, counts: &[f32], cfg: &A1Config, steps: u32) -> f64 {
    let mut solver = TransmissionEm::new(system, counts, cfg.clone());
    let ctl = Control::new(steps, 1);
    for _ in 0..steps {
        if solver.step(&ctl).converged {
            break;
        }
    }
    black_box(solver.estimate());
    solver.objective()
}

fn time<T>(runs: u32, mut body: impl FnMut() -> T) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let start = Instant::now();
        black_box(body());
        samples.push(start.elapsed());
    }
    samples
}

fn report(name: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let n = samples.len();
    let min = samples[0];
    let median = samples[n / 2];
    let mean = samples.iter().sum::<Duration>() / n as u32;
    println!(
        "{name:<28} min {:>8.3} ms   median {:>8.3} ms   mean {:>8.3} ms   ({n} runs)",
        min.as_secs_f64() * 1.0e3,
        median.as_secs_f64() * 1.0e3,
        mean.as_secs_f64() * 1.0e3,
    );
}

fn main() {
    let (system, counts, cfg) = dataset();
    println!(
        "a1 on the default scene at L0: {} rays, {} voxels, {} subsets",
        system.ray_count(),
        system.voxel_count(),
        cfg.subsets
    );

    // Warm-up: the first runs pay for page faults and the allocator, which has
    // nothing to do with the algorithm.
    for _ in 0..5 {
        black_box(reconstruct(&system, &counts, &cfg, 20));
    }

    report(
        "a1_one_subset_step",
        time(200, || one_step(&system, &counts, &cfg)),
    );

    let final_objective = reconstruct(&system, &counts, &cfg, 20);
    println!("a1 reconstruction after 20 steps: NLL {final_objective:.6}");
    report(
        "a1_reconstruction_20_steps",
        time(20, || reconstruct(&system, &counts, &cfg, 20)),
    );
}
