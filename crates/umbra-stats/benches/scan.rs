//! A6 scan cost: 10⁶ candidates in 2-D and in 3-D (spec §5.11's budget row).
//!
//! The criterion this measures is "1e6 candidates under 2 s in 3-D and 50 ms in
//! 2-D". Both systems here are built with **parallel rays through every voxel
//! along one axis**, which makes each voxel's footprint small and is the
//! friendliest geometry the scan can meet. A real cone-beam scene spreads the
//! same opacity over more rays, so treat the numbers as a lower bound: they
//! answer "is the kernel's constant factor in the right decade", not "is every
//! 3-D experiment inside 2 s".
//!
//! Run with `cargo bench -p umbra-stats --bench scan`.

use std::time::{Duration, Instant};

use umbra_core::{Csr, Grid2, Grid3, GridAny};
use umbra_forward::{RayCell, System};
use umbra_stats::{scan, scan_with_map, z_and_alpha_map, ScanConfig, ScanProblem};

/// Two-dimensional scene: one ray per row, `n` cells long.
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
    finish(Csr::new(indptr, indices, data), GridAny::Two(grid), n)
}

/// Three-dimensional scene: one ray along `z` for each `(x, y)` column.
fn scene_3d(n: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let grid = Grid3::new([n as u32, n as u32, n as u32], [0.0; 3], 0.01);
    let plane = n * n;
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for column in 0..plane {
        for k in 0..n {
            indices.push((k * plane + column) as u32);
            data.push(1.0f32);
        }
        indptr.push(indices.len() as u32);
    }
    finish(Csr::new(indptr, indices, data), GridAny::Three(grid), plane)
}

fn finish(l: Csr<f32>, grid: GridAny, rays: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut density = vec![2.65f32; grid.cell_count()];
    // A buried inclusion, so the scan has something to find and the amplitude
    // path is exercised rather than only the H0 arithmetic.
    let voxels = grid.cell_count();
    let start = voxels / 3;
    let end = start + voxels / 50;
    density[start..end].fill(1.0);
    let system = System {
        l,
        rays: (0..rays)
            .map(|_| RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0,
            })
            .collect(),
        grid,
    };
    let opacity = system.opacity(&density);
    let lambda0: Vec<f32> = opacity
        .iter()
        .map(|&x| 1.0e3 * (-x / 1.0e3).exp())
        .collect();
    let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1.0e3).collect();
    (system, density, lambda0, dlambda_dx)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1.0e3
}

/// Repetitions per measurement; the fastest is reported, so more runs only help.
///
/// `UMBRA_SCAN_RUNS` raises it. On a busy machine the minimum over 25 runs is a
/// far better estimate of the kernel's cost than the minimum over 7, because it
/// is more likely to catch a quiet stretch.
fn runs() -> usize {
    std::env::var("UMBRA_SCAN_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(7)
}

fn main() {
    // One scale is exactly the "10⁶ candidates" the budget row names; the second
    // line reports what a multi-scale bank costs on the same scene, because a
    // real scan runs a bank and the budget does not say so.
    let single = ScanConfig {
        scales_voxels: vec![1.0],
        top_k: 10,
        ..ScanConfig::default()
    };
    let multi = ScanConfig {
        scales_voxels: vec![1.0, 2.0],
        ..single.clone()
    };

    let runs = runs();
    for (label, (system, density, lambda0, dlambda_dx), budget) in [
        ("2-D", scene_2d(1000), 50.0f64),
        ("3-D", scene_3d(100), 2000.0f64),
    ] {
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.02).collect();
        let problem = ScanProblem {
            system: &system,
            counts: &counts,
            lambda0: &lambda0,
            dlambda_dx: &dlambda_dx,
            density: &density,
            attenuation_gcm2: 1.0e3,
        };
        for (name, cfg) in [("one scale", &single), ("two scales", &multi)] {
            // One warm-up so the first allocation does not land in the measurement.
            let _ = scan(&problem, cfg);
            let mut samples = Vec::new();
            let mut correlations = Vec::new();
            let mut peak = 0.0f32;
            for _ in 0..runs {
                let started = Instant::now();
                let result = scan(&problem, cfg);
                samples.push(started.elapsed());
                correlations.push(result.timing.correlate);
                peak = result.peak.map_or(0.0, |c| c.z);
            }
            samples.sort();
            correlations.sort();
            // The best run is the one that saw the least contention. On a shared
            // machine the minimum estimates the kernel's cost and the median
            // reports whatever else the scheduler was doing, which is how the
            // same binary reads 160 ms one minute and 460 ms the next.
            let total = ms(samples[0]);
            let correlation = ms(correlations[0]);
            let typical = ms(correlations[correlations.len() / 2]);
            println!(
                "{label} {name}: {} candidates, {} rays, {} voxels — best of {runs}: {total:>7.1} ms \
                 total, {correlation:>7.1} ms correlation of {:>5.0} ms budget (median correlation \
                 {typical:.1} ms, peak Z {peak:.2})",
                system.voxel_count() * cfg.scales_voxels.len(),
                system.ray_count(),
                system.voxel_count(),
                budget,
            );
        }
    }

    // The app asks for the peak and the field together. Doing that as two calls
    // walked the candidate set twice, which was the whole cost of the browser's
    // SCAN stage; this reports both shapes on the same 10⁶-candidate scene.
    {
        let (system, density, lambda0, dlambda_dx) = scene_2d(1000);
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.02).collect();
        let problem = ScanProblem {
            system: &system,
            counts: &counts,
            lambda0: &lambda0,
            dlambda_dx: &dlambda_dx,
            density: &density,
            attenuation_gcm2: 1.0e3,
        };
        let _ = scan_with_map(&problem, &single);
        let mut split = Vec::new();
        let mut joined = Vec::new();
        for _ in 0..runs {
            let started = Instant::now();
            let _ = scan(&problem, &single);
            let _ = z_and_alpha_map(&problem, &single);
            split.push(started.elapsed());
            let started = Instant::now();
            let _ = scan_with_map(&problem, &single);
            joined.push(started.elapsed());
        }
        split.sort();
        joined.sort();
        println!(
            "2-D peak + field: {:.1} ms as two sweeps, {:.1} ms as one (best of {runs})",
            ms(split[0]),
            ms(joined[0]),
        );
    }
}
