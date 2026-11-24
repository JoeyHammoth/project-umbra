//! Recovery through the ray tracer.
//!
//! `lee.rs`'s recovery test uses a hand-built two-view operator, so the scan and
//! the correction are isolated from the geometry. This asks the opposite
//! question: the same claim on the operator `build_system_2d` actually produces —
//! exact DDA path lengths, partial volume where a ray clips a voxel, four
//! detectors with a fan each. A test that only works on hand-written matrices
//! would be hiding whatever the tracer does differently.
//!
//! The forward model here is L0 with the scene's own attenuation length, applied
//! to the traced opacity. It is not the flux-table model the app ships; what is
//! under test is that the scan finds a void in traced geometry, and using one law
//! for both the counts and the derivative keeps that question clean.

use umbra_core::Grid2;
use umbra_forward::{build_system_2d, Scene2D, System};
use umbra_physics::{AngularBinning, Detector, Pose};
use umbra_stats::{scan, ScanConfig, ScanProblem};

/// Short enough that a 32 cm slab is a few attenuation lengths, so a void in it
/// is a real object rather than a percent of one.
const ATTENUATION_GCM2: f32 = 20.0;

/// Scale of the expected counts: counts per ray at zero opacity.
const BRIGHTNESS: f32 = 1.0e4;

const N: usize = 32;
const CENTRE: usize = 16;
const RADIUS_VOXELS: f32 = 3.5;

fn detector(position: [f32; 2], centre: f32) -> Detector {
    Detector {
        pose: Pose {
            position: [position[0], position[1], 0.0],
            normal: [0.0, 1.0, 0.0],
            up: [1.0, 0.0, 0.0],
        },
        area_m2: 1.0,
        eff: 1.0,
        // ±0.7 rad from the inward direction: wide enough that every ray crosses
        // the grid. A ray that missed the scene would carry λ⁰ with no material
        // under it, which is a bright, empty ray the Fisher term would trust.
        binning: AngularBinning::Flat2D {
            min: centre - 0.7,
            max: centre + 0.7,
            bins: 9,
        },
        sigma_ang_mrad: 0.0,
        bg_rate_hz: 0.0,
    }
}

/// A four-view scene: below looking up, above looking down, and one each side.
///
/// Angular diversity is the point. A single parallel view cannot localise along
/// its own rays — the scan measured an identical `Z` at every x when the fixture
/// had one — so the geometry has to constrain both axes before "where" means
/// anything.
fn scene() -> (System, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let grid = Grid2::new([N as u32, N as u32], [0.0, 0.0], 0.01);
    let system = build_system_2d(&Scene2D::new(
        grid,
        vec![
            detector([0.16, -0.02], 0.0),
            detector([0.16, 0.34], std::f32::consts::PI),
            detector([-0.02, 0.16], std::f32::consts::FRAC_PI_2),
            detector([0.34, 0.16], -std::f32::consts::FRAC_PI_2),
        ],
    ));

    let density = vec![2.65f32; system.voxel_count()];
    let mut voided = density.clone();
    for j in 0..N {
        for i in 0..N {
            let (dx, dy) = (i as f32 - CENTRE as f32, j as f32 - CENTRE as f32);
            if dx * dx + dy * dy <= RADIUS_VOXELS * RADIUS_VOXELS {
                voided[j * N + i] = 0.0;
            }
        }
    }

    let counts_at = |density: &[f32]| -> Vec<f32> {
        system
            .opacity(density)
            .iter()
            .map(|&x| BRIGHTNESS * (-x / ATTENUATION_GCM2).exp())
            .collect()
    };
    let lambda0 = counts_at(&density);
    let lambda1 = counts_at(&voided);
    let dlambda_dx: Vec<f32> = lambda0
        .iter()
        .map(|&lambda| -lambda / ATTENUATION_GCM2)
        .collect();

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let counts: Vec<f32> = lambda1
        .iter()
        .map(|&lambda| rng.poisson(f64::from(lambda)) as f32)
        .collect();
    (system, density, lambda0, counts, dlambda_dx)
}

#[test]
fn a_void_planted_in_a_traced_scene_is_found_where_it_was_placed() {
    let (system, density, lambda0, counts, dlambda_dx) = scene();
    assert!(
        system.ray_count() >= 30,
        "thirty-six rays, not a toy geometry: {}",
        system.ray_count()
    );

    let problem = ScanProblem {
        system: &system,
        counts: &counts,
        lambda0: &lambda0,
        dlambda_dx: &dlambda_dx,
        density: &density,
        attenuation_gcm2: ATTENUATION_GCM2,
    };
    let cfg = ScanConfig {
        scales_voxels: vec![1.0, 2.0, 3.0],
        top_k: 5,
        ..ScanConfig::default()
    };
    let peak = scan(&problem, &cfg)
        .peak
        .expect("a full void across four views is not nothing");

    let (px, py) = (peak.voxel % N, peak.voxel / N);
    let (dx, dy) = (px as f64 - CENTRE as f64, py as f64 - CENTRE as f64);
    let offset = (dx * dx + dy * dy).sqrt();
    println!(
        "traced scene: {} rays, peak ({px}, {py}), off {offset:.2} vox, Z {:.2}, depth {:.2}, scale {}",
        system.ray_count(),
        peak.z,
        peak.alpha_hat,
        peak.scale
    );
    assert!(
        offset <= f64::from(RADIUS_VOXELS),
        "the peak landed at ({px}, {py}), {offset:.1} voxels from the planted ({CENTRE}, {CENTRE}) \
         — outside the void it was given"
    );
    assert!(
        peak.z > 5.0,
        "a full void in four views should not be marginal: Z = {}",
        peak.z
    );
    assert!(
        peak.alpha_hat > 0.3,
        "the fitted depth {} should be a real fraction of the planted 1.0",
        peak.alpha_hat
    );
}

/// SplitMix64 and a Knuth Poisson draw; the counts here are a few hundred at
/// most, so the sampling loop is short and the point is reproducibility.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn poisson(&mut self, lambda: f64) -> u32 {
        if !lambda.is_finite() || lambda <= 0.0 {
            return 0;
        }
        let limit = (-lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= self.uniform();
            if p <= limit {
                return k - 1;
            }
        }
    }
}
