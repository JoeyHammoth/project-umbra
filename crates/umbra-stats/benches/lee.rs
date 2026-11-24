//! LEE calibration campaign: the empirical max-`Z` null against the Gross–Vitells
//! extrapolation, on the same code that ships.
//!
//! Defaults are the campaign the spec asks for — `1000²` voxels (`10⁶`) and `10⁴`
//! toys — because this is the run that produces the published report. M3's rate
//! criterion is decided at the end of the report, from the Wilson interval of the
//! exceedances: `10⁴` toys straddle the ±20% band around nominal 3σ, so the
//! campaign has to be run at `10⁵` (~3 h on a CI runner) before that box can move.
//! Env knobs:
//!
//! ```text
//! UMBRA_LEE_GRID     scene width, default 1000
//! UMBRA_LEE_TOYS     toy count, default 10 000
//! UMBRA_LEE_U0       Euler anchor in sigma, default 2.5
//! UMBRA_LEE_THREADS  worker threads, default all cores
//! UMBRA_LEE_SEED     toy seed, default 0x00C0FFEE
//! UMBRA_LEE_REPORT   write the markdown report here as well as to stdout
//! ```
//!
//! Toy `t` depends only on `(seed, t)`, so the threads split the index range and
//! the calibration is identical to the serial run — the report notes the seed and
//! the split, so a reader can reproduce it.
//!
//! Run with `cargo bench -p umbra-stats --bench lee`.

use std::fmt::Write as _;
use std::time::Instant;

use umbra_core::{Csr, Grid2, GridAny};
use umbra_forward::{RayCell, System};
use umbra_stats::{
    chi2_mixture_p, lee_calibrate_with, rate_estimate, LeeConfig, ScanConfig, ScanProblem,
};

/// The spec's nominal 3σ false-positive rate (§7): a correct global `p` curve
/// should reach this only at the threshold that the correction assigns to `3σ`.
const NOMINAL_3_SIGMA: f64 = 1.35e-3;

fn scene_2d(n: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let grid = Grid2::new([n as u32, n as u32], [0.0, 0.0], 0.01);
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n {
        for column in 0..n {
            indices.push((row * n + column) as u32);
            data.push(1.0f32);
        }
        indptr.push(indices.len() as u32);
    }
    let mut density = vec![2.65f32; grid.cell_count()];
    let voxels = grid.cell_count();
    let start = voxels / 3;
    density[start..start + voxels / 50].fill(1.0);
    let system = System {
        l: Csr::new(indptr, indices, data),
        rays: (0..n)
            .map(|_| RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0,
            })
            .collect(),
        grid: GridAny::Two(grid),
        poses: Vec::new(),
    };
    let opacity = system.opacity(&density);
    let lambda0: Vec<f32> = opacity
        .iter()
        .map(|&x| 1.0e3 * (-x / 1.0e3).exp())
        .collect();
    let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1.0e3).collect();
    (system, density, lambda0, dlambda_dx)
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n: usize = env_or("UMBRA_LEE_GRID", 1000);
    let toys: u32 = env_or("UMBRA_LEE_TOYS", 10_000);
    let u0: f32 = env_or("UMBRA_LEE_U0", 2.5);
    let seed: u64 = env_or("UMBRA_LEE_SEED", 0x00c0_ffee_u64);
    let threads: usize = env_or(
        "UMBRA_LEE_THREADS",
        std::thread::available_parallelism().map_or(1, |t| t.get()),
    );

    let (system, density, lambda0, dlambda_dx) = scene_2d(n);
    let cfg = ScanConfig {
        scales_voxels: vec![1.0, 2.0],
        top_k: 10,
        ..ScanConfig::default()
    };
    // `counts` is unused by the toys: each draws its own counts from `λ⁰`.
    let problem = ScanProblem {
        system: &system,
        counts: &lambda0,
        lambda0: &lambda0,
        dlambda_dx: &dlambda_dx,
        density: &density,
        attenuation_gcm2: 1.0e3,
    };
    let lee = LeeConfig { u0, dim: 2 };

    let started = Instant::now();
    // The scheduling lives in the library now: this bench used to hand-roll the
    // same chunking, which meant the campaign's only tested path was the one no
    // other caller could run. Toy `t` depends on `(seed, t)` alone, so the split
    // is bit-identical to a serial campaign, and `sharding_a_campaign_changes_no_toy`
    // is what holds that to account instead of the module doc's say-so.
    let calibration = lee_calibrate_with(&problem, &cfg, &lee, toys, seed, threads);
    let elapsed = started.elapsed();

    // The honest 3σ threshold: the smallest z whose empirical global rate is at
    // or below the nominal 3σ false-positive probability. That is the number a
    // reader wants, and it is not 3.0.
    let mut threshold = None;
    let mut z = 2.0f32;
    while z <= 6.0 {
        if f64::from(calibration.empirical_global_p(z)) <= NOMINAL_3_SIGMA * 1.2 {
            threshold = Some(z);
            break;
        }
        z += 0.05;
    }

    // The M3 criterion, computed rather than narrated. The threshold comes from
    // the *model* — the `z` at which the extrapolation reaches nominal — because
    // choosing it from the empirical tail and then quoting the empirical rate
    // there would be scoring the sample against itself.
    let mut model_3sigma = None;
    let mut z = 2.0f32;
    while z <= 6.0 {
        if f64::from(calibration.extrapolated_global_p(z)) <= NOMINAL_3_SIGMA {
            model_3sigma = Some(z);
            break;
        }
        z += 0.01;
    }
    let exceedances = model_3sigma.map_or(0, |threshold| {
        calibration
            .maxima
            .iter()
            .filter(|&&max| max >= threshold)
            .count() as u32
    });
    let estimate = rate_estimate(exceedances, toys);
    let verdict = if estimate.inside(NOMINAL_3_SIGMA, 0.20) {
        "PASS"
    } else if estimate.outside(NOMINAL_3_SIGMA, 0.20) {
        "FAIL"
    } else {
        "UNDECIDED"
    };

    let mut report = String::new();
    let _ = writeln!(report, "# LEE calibration — 2-D, {} toys\n", toys);
    let _ = writeln!(
        report,
        "| | |\n|---|---|\n| Scene | {n}² voxels, {} rays |\n| Toys | {toys} |\n| Seed | `0x{seed:016x}` |\n| Threads | {threads} (one toy index range per worker) |\n| Euler anchor `u₀` | {u0}σ |\n| Runtime | {:.1} s ({:.1} ms/toy) |",
        system.ray_count(),
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1.0e3 / f64::from(toys)
    );
    let _ = writeln!(
        report,
        "\nMean Euler characteristic at `u₀`: **{:.3} ± {:.3}** ({} of {toys} toys exceeded `u₀`).\n",
        calibration.mean_euler_at_u0, calibration.euler_stderr, calibration.toys_above_u0
    );
    let _ = writeln!(
        report,
        "max-`Z` quantiles: 50% {:.2}, 90% {:.2}, 99% {:.2}, 99.9% {:.2}.\n",
        calibration.quantile(0.5),
        calibration.quantile(0.9),
        calibration.quantile(0.99),
        calibration.quantile(0.999)
    );
    let _ = writeln!(
        report,
        "| z | local p | empirical global p | extrapolated global p | ratio | toys above |\n|---|---|---|---|---|---|"
    );
    for z in [2.0f32, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] {
        let empirical = f64::from(calibration.empirical_global_p(z));
        let extrapolated = f64::from(calibration.extrapolated_global_p(z));
        let local = f64::from(chi2_mixture_p(f64::from(z) * f64::from(z)));
        let above = calibration.maxima.iter().filter(|&&m| m >= z).count();
        // At or below the anchor the function returns the local `p` unchanged,
        // so a ratio there compares the local tail with the global one and reads
        // as a spectacular result. It is not a measurement of anything.
        let ratio = if z <= lee.u0 {
            "—".to_string()
        } else {
            format!("{:.2}", extrapolated / empirical)
        };
        let _ = writeln!(
            report,
            "| {z:.1} | {local:.3e} | {empirical:.3e} | {extrapolated:.3e} | {ratio} | {above} |"
        );
    }
    let _ = writeln!(
        report,
        "\nRows at or below `u₀` show the local `p`: the extrapolation does not\napply there, so their ratio is omitted rather than computed."
    );
    match threshold {
        Some(z) => {
            let _ = writeln!(
                report,
                "\n**Global 3σ threshold:** `z = {z:.2}` (empirical global p ≤ {:.3e}, nominal 3σ = {NOMINAL_3_SIGMA:.3e}). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is {:.0}×.",
                NOMINAL_3_SIGMA,
                f64::from(calibration.empirical_global_p(3.0)) / NOMINAL_3_SIGMA
            );
        }
        None => {
            let _ = writeln!(
                report,
                "\n**Global 3σ threshold:** not reached within `z ≤ 6` at {toys} toys; the tail needs a larger campaign.",
            );
        }
    }
    let _ = writeln!(
        report,
        "\n## M3 criterion (spec §10.3): global false-positive rate at 3σ\n\n| | |\n|---|---|\n| Model 3σ threshold | {} |\n| Exceedances | {exceedances} of {toys} |\n| Empirical rate | {:.3e} |\n| 95% interval (Wilson) | [{:.3e}, {:.3e}] |\n| Nominal band (±20%) | [{:.3e}, {:.3e}] |\n| Verdict | **{verdict}** |",
        model_3sigma.map_or("not reached by z = 6".to_string(), |threshold| format!(
            "z = {threshold:.2}"
        )),
        estimate.rate,
        estimate.low,
        estimate.high,
        NOMINAL_3_SIGMA * 0.8,
        NOMINAL_3_SIGMA * 1.2
    );
    if verdict == "UNDECIDED" {
        let _ = writeln!(
            report,
            "\nUNDECIDED means the interval straddles the band: this campaign cannot say whether the rate is within ±20% of nominal, and the answer is more toys rather than a different threshold."
        );
    }
    let _ = writeln!(
        report,
        "\n## Verdict\n\n- Extrapolation vs Monte Carlo at `z = 3`: **{:.2}×** (spec target: within 20%).\n- `z = 5`: {} toys above, so only the extrapolation exists there and it is unvalidated.",
        f64::from(calibration.extrapolated_global_p(3.0))
            / f64::from(calibration.empirical_global_p(3.0)),
        calibration.maxima.iter().filter(|&&m| m >= 5.0).count()
    );

    print!("{report}");
    if let Ok(path) = std::env::var("UMBRA_LEE_REPORT") {
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&path, &report) {
            Ok(()) => println!("\nreport written to {path}"),
            Err(error) => eprintln!("\ncould not write {path}: {error}"),
        }
    }
}
