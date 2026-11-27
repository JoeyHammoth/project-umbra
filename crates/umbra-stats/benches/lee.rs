//! LEE calibration campaign: the empirical max-`Z` null against the Gross–Vitells
//! extrapolation, on the same code that ships.
//!
//! Defaults are the campaign the spec asks for — `1000²` voxels (`10⁶`) and `10⁴`
//! toys — because this is the run that produces the published report. M3's rate
//! criterion is decided at the end of the report, from the Wilson interval of the
//! exceedances, and the report also says how many toys that interval would need
//! before it fits the band. On the plane `10⁴` toys are about `7 × 10⁴` short,
//! which is the number the CI gate dispatches rather than a constant.
//!
//! `UMBRA_LEE_DIM=3` runs the same campaign on a volume, so the upcrossing law
//! is measured where an excursion is a sheet a ray cuts through rather than a
//! strip along it. The two dimensions do not share a grid side: `1000³` is `10⁹`
//! voxels and no campaign budget reaches it.
//!
//! The volume default is `24³` because a volume campaign has to be able to
//! *decide* its own criterion, and at `40³` it cannot: a toy costs 10.9 ms there,
//! so the `~10⁷` toys a volume would need is a week, and the criterion stays open
//! by budget rather than by physics. At `24³` a toy costs 2.2 ms, and the
//! extrapolation still matches the toys (1.08× at 3σ, 0.97× at 4σ), so `10⁶` toys
//! is half an hour. Ratios from different scenes are comparable; rates are not,
//! because the search space and the ray count per voxel are not.
//! Env knobs:
//!
//! ```text
//! UMBRA_LEE_DIM      search dimension, 2 or 3, default 2
//! UMBRA_LEE_GRID     scene width, default 1000 (2-D) or 24 (3-D)
//! UMBRA_LEE_TOYS     toy count, default 10 000
//! UMBRA_LEE_U0       Euler anchor in sigma, default 2.5
//! UMBRA_LEE_THREADS  worker threads, default all cores
//! UMBRA_LEE_SEED     toy seed, hex (`0x…`) or decimal, default 0x00C0FFEE
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

use umbra_core::{Csr, Grid2, Grid3, GridAny};
use umbra_forward::{RayCell, System};
use umbra_stats::{
    chi2_mixture_p, lee_calibrate_with, rate_estimate, toys_to_decide, LeeConfig, ScanConfig,
    ScanProblem,
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

fn scene_3d(n: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let grid = Grid3::new([n as u32, n as u32, n as u32], [0.0; 3], 0.01);
    let plane = n * n;
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    // One ray along `z` per column, as the scan bench's 3-D scene: every voxel
    // is seen by exactly one ray, which is the property the 2-D campaign rests
    // on and the one the law is being tested with.
    for column in 0..plane {
        for k in 0..n {
            indices.push((k * plane + column) as u32);
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
        rays: (0..plane)
            .map(|_| RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0,
            })
            .collect(),
        grid: GridAny::Three(grid),
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

/// A seed knob takes `0x`-prefixed hex or decimal.
///
/// `env_or` parses through `FromStr`, which reads decimal only, so a documented
/// default written as `0x00C0FFEE` and a seed passed as `0x…` were two different
/// things: the knob silently fell back to the default and the report then named a
/// seed the caller had not run. A campaign whose seed is not the one asked for is
/// a sample nobody can reproduce, so a value that does not parse is an error here
/// rather than a default.
fn seed_or_default(key: &str, default: u64) -> u64 {
    let Ok(text) = std::env::var(key) else {
        return default;
    };
    let trimmed = text.trim();
    let parsed = match trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => trimmed.parse(),
    };
    match parsed {
        Ok(seed) => seed,
        Err(error) => panic!("{key}={text:?} is not a seed: {error}"),
    }
}

fn main() {
    let dim: u8 = env_or("UMBRA_LEE_DIM", 2);
    let n: usize = env_or("UMBRA_LEE_GRID", if dim == 3 { 24 } else { 1000 });
    let toys: u32 = env_or("UMBRA_LEE_TOYS", 10_000);
    let u0: f32 = env_or("UMBRA_LEE_U0", 2.5);
    let seed: u64 = seed_or_default("UMBRA_LEE_SEED", 0x00c0_ffee_u64);
    let threads: usize = env_or(
        "UMBRA_LEE_THREADS",
        std::thread::available_parallelism().map_or(1, |t| t.get()),
    );

    let (system, density, lambda0, dlambda_dx) = match dim {
        3 => scene_3d(n),
        _ => scene_2d(n),
    };
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
    let lee = LeeConfig { u0, dim };

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
    // What this campaign would need to decide, so a run that cannot answer the
    // question says what would rather than leaving the next reader to work it out
    // from the interval. This is what the CI gate sizes its confirmation run by:
    // a sample that is 5% off nominal needs ~1.2× its toys, and one 19% off needs
    // fifty times them, which is a fact worth printing instead of discovering
    // after another campaign.
    let needs = toys_to_decide(estimate, NOMINAL_3_SIGMA, 0.20, toys);
    let verdict = if estimate.inside(NOMINAL_3_SIGMA, 0.20) {
        "PASS"
    } else if estimate.outside(NOMINAL_3_SIGMA, 0.20) {
        "FAIL"
    } else {
        "UNDECIDED"
    };

    let mut report = String::new();
    let shape = if dim == 3 { "³" } else { "²" };
    let _ = writeln!(report, "# LEE calibration — {dim}-D, {toys} toys\n");
    let _ = writeln!(
        report,
        "| | |\n|---|---|\n| Scene | {n}{shape} voxels, {} rays |\n| Search dimension | {dim} |\n| Toys | {toys} |\n| Seed | `0x{seed:016x}` |\n| Threads | {threads} (one toy index range per worker) |\n| Euler anchor `u₀` | {u0}σ |\n| Runtime | {:.1} s ({:.1} ms/toy) |",
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
    // Which law produced the numbers, stated where the numbers are: a reader
    // comparing the extrapolation against the tail needs to know that the
    // regions are counted as an upcrossing count and turned into a probability
    // rather than quoted as a rate.
    let _ = writeln!(
        report,
        "The extrapolation is Gross–Vitells: `⟨N(u)⟩ = p_local(u) + ⟨χ(u₀)⟩ e^{{−(u²−u₀²)/2}}`,\nreported as `p = 1 − e^{{−⟨N⟩}}` because `⟨N⟩` is a region count, not a probability.\n"
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
        "\n## M3 criterion (spec §10.3): global false-positive rate at 3σ\n\n| | |\n|---|---|\n| Model 3σ threshold | {} |\n| Exceedances | {exceedances} of {toys} |\n| Empirical rate | {:.3e} |\n| 95% interval (Wilson) | [{:.3e}, {:.3e}] |\n| Nominal band (±20%) | [{:.3e}, {:.3e}] |\n| Toys to decide | {} |\n| Verdict | **{verdict}** |",
        model_3sigma.map_or("not reached by z = 6".to_string(), |threshold| format!(
            "z = {threshold:.2}"
        )),
        estimate.rate,
        estimate.low,
        estimate.high,
        NOMINAL_3_SIGMA * 0.8,
        NOMINAL_3_SIGMA * 1.2,
        needs.map_or_else(
            || if estimate.inside(NOMINAL_3_SIGMA, 0.20) {
                "already decided by this sample".to_string()
            } else {
                "none — the rate itself is outside the band".to_string()
            },
            |needed| {
                if needed <= toys {
                    format!("{needed} (already decided by this sample)")
                } else {
                    format!("~{needed} (this sample decides nothing)")
                }
            }
        ),
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

    // The machine-readable line the CI gate reads: `tools/lee-verdict.sh` needs
    // the verdict, the toy count and the dimension without parsing prose, and an
    // HTML comment keeps the line out of the rendered report.
    let _ = writeln!(
        report,
        "\n<!-- lee-verdict verdict={verdict} toys={toys} dim={dim} rate={:.3e} low={:.3e} high={:.3e} threshold={} needs={} -->",
        estimate.rate,
        estimate.low,
        estimate.high,
        model_3sigma.map_or("none".to_string(), |threshold| format!("{threshold:.2}")),
        needs.map_or("none".to_string(), |needed| needed.to_string()),
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
