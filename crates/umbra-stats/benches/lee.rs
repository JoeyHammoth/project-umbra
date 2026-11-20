//! LEE calibration run: the empirical max-`Z` null against the Gross–Vitells
//! extrapolation, on the same code that ships.
//!
//! `UMBRA_LEE_GRID` sets the scene width (default 400) and `UMBRA_LEE_TOYS` the
//! toy count (default 500). The spec's `3σ` calibration target — global rate
//! within ±20% of nominal — needs `10⁴` toys on the full `10⁶`-voxel scene, so
//! that campaign is a parameter change here rather than a rewrite. At the default
//! size it finishes in seconds, which is the point: the estimator's agreement
//! with its own Monte Carlo tail is checkable on every commit.
//!
//! Run with `cargo bench -p umbra-stats --bench lee`.

use std::time::Instant;

use umbra_core::{Csr, Grid2, GridAny};
use umbra_forward::{RayCell, System};
use umbra_stats::{lee_calibrate, LeeConfig, ScanConfig, ScanProblem};

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
    };
    let opacity = system.opacity(&density);
    let lambda0: Vec<f32> = opacity
        .iter()
        .map(|&x| 1.0e3 * (-x / 1.0e3).exp())
        .collect();
    let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1.0e3).collect();
    (system, density, lambda0, dlambda_dx)
}

fn main() {
    let n: usize = std::env::var("UMBRA_LEE_GRID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let toys: u32 = std::env::var("UMBRA_LEE_TOYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
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
    let u0: f32 = std::env::var("UMBRA_LEE_U0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.5);
    let lee = LeeConfig { u0, dim: 2 };

    let started = Instant::now();
    let calibration = lee_calibrate(&problem, &cfg, &lee, toys, 0x00c0_ffee);
    let elapsed = started.elapsed();

    println!(
        "{n}² voxels, {} rays, {toys} toys in {:.2} s ({:.1} ms/toy)",
        system.ray_count(),
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1.0e3 / f64::from(toys)
    );
    println!(
        "mean χ(u₀ = {:.1}) = {:.3} ± {:.3}, {:} toys above u₀",
        lee.u0, calibration.mean_euler_at_u0, calibration.euler_stderr, calibration.toys_above_u0
    );
    println!(
        "max-Z quantiles: 50% {:.2}, 90% {:.2}, 99% {:.2}",
        calibration.quantile(0.5),
        calibration.quantile(0.9),
        calibration.quantile(0.99)
    );
    println!();
    println!("  z    local p    empirical global    extrapolated    toys above");
    for z in [1.0f32, 2.0, 3.0, 4.0, 5.0] {
        let empirical = calibration.empirical_global_p(z);
        let extrapolated = calibration.extrapolated_global_p(z);
        let above = calibration.maxima.iter().filter(|&&m| m >= z).count();
        println!(
            "{z:>4.1}  {:.3e}      {:.3e}           {:.3e}        {above:>4}",
            f64::from(umbra_stats::chi2_mixture_p(f64::from(z) * f64::from(z))),
            empirical,
            extrapolated
        );
    }
}
