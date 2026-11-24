//! System assembly: detectors and angular bins → rays → CSR path lengths.
//!
//! Spec §5.3 requires row-per-ray assembly with deterministic ordering. That is
//! what makes the content-hash cache keys in §4.5 meaningful: if assembly order
//! depended on scheduling, the same scene would hash to different systems.

use umbra_core::{Csr, Grid2, GridAny};
use umbra_physics::{AngularBinning, Detector};

use crate::dda::traverse_2d;
use crate::system::{RayCell, System};

/// A 2-D slice scene: a grid, the detectors looking into it, and how far a ray
/// may travel.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene2D {
    /// Voxel grid, in metres.
    pub grid: Grid2,
    /// Detectors, in scene order. Order is part of the system identity.
    pub detectors: Vec<Detector>,
    /// Maximum ray length, in metres. Usually the scene's diagonal.
    pub max_path_m: f32,
}

impl Scene2D {
    /// A scene with a ray budget derived from the grid diagonal.
    pub fn new(grid: Grid2, detectors: Vec<Detector>) -> Self {
        let width = grid.n[0] as f32 * grid.h_m;
        let height = grid.n[1] as f32 * grid.h_m;
        Self {
            grid,
            detectors,
            max_path_m: (width * width + height * height).sqrt(),
        }
    }
}

/// Builds the path-length operator for a slice scene.
///
/// # Conventions
///
/// - A `Flat2D` bin with centre angle `φ` (radians, measured from straight up,
///   positive toward `+x`) produces the direction `(sin φ, cos φ)`, so the zenith
///   cosine is `cos φ` and `k_scale` can include it.
/// - Path lengths are stored in **centimetres**, the unit `ρ` in g cm⁻³ needs to
///   produce opacity in g cm⁻².
/// - A detector with non-positive efficiency contributes rays with zero
///   acceptance rather than being dropped, so ray indices stay stable when the
///   user edits an efficiency.
pub fn build_system_2d(scene: &Scene2D) -> System {
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    let mut rays = Vec::new();

    for (det_index, detector) in scene.detectors.iter().enumerate() {
        let origin = [detector.pose.position[0], detector.pose.position[1]];
        let bin_angles: Vec<(f32, f32)> = match detector.binning {
            AngularBinning::Flat2D { min, max, bins } => {
                let bins = bins.max(1);
                let width = (max - min) / bins as f32;
                (0..bins)
                    .map(|k| (min + width * (k as f32 + 0.5), width))
                    .collect()
            }
            // A 3-D binning used in slice mode is projected onto the zenith
            // fan: the azimuth axis collapses, and the zenith rows become bins.
            AngularBinning::AzZen {
                zen_min,
                zen_max,
                zen_bins,
                az_bins,
                ..
            } => {
                let bins = zen_bins.max(1);
                let width = (zen_max - zen_min) / bins as f32;
                let azimuth_weight = 1.0 / az_bins.max(1) as f32;
                (0..bins)
                    .map(|k| (zen_min + width * (k as f32 + 0.5), width * azimuth_weight))
                    .collect()
            }
        };

        for (phi, delta_phi) in bin_angles {
            let direction = [phi.sin(), phi.cos()];
            // Zenith cosine for the fan: straight up is φ = 0.
            let cos_zenith = phi.cos().max(0.0);
            let area_cm2 = detector.area_m2 * 1.0e4;
            let k_scale = detector.eff * area_cm2 * cos_zenith * delta_phi;

            let hits = traverse_2d(&scene.grid, origin, direction, scene.max_path_m);
            for hit in hits {
                indices.push(hit.index as u32);
                data.push(hit.length_m * 100.0); // metres → centimetres
            }
            indptr.push(indices.len() as u32);
            rays.push(RayCell {
                det: det_index as u16,
                zen: phi,
                az: 0.0,
                k_scale,
            });
        }
    }

    let l = Csr::new(indptr, indices, data);
    debug_assert_eq!(l.nrows(), rays.len());
    System {
        l,
        rays,
        grid: GridAny::Two(scene.grid.clone()),
        // One pose per detector, in scene order, which is what `RayCell::det`
        // indexes. Assembly order already has to be deterministic (spec §5.3),
        // so this is the same ordering guarantee applied to the geometry.
        poses: scene
            .detectors
            .iter()
            .map(|detector| detector.pose)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{dot_product_test, Primitive, Quat};
    use umbra_physics::Pose;

    fn detector(x: f32, y: f32, bins: u32) -> Detector {
        Detector {
            pose: Pose {
                position: [x, y, 0.0],
                normal: [0.0, 1.0, 0.0],
                up: [1.0, 0.0, 0.0],
            },
            area_m2: 1.0,
            eff: 1.0,
            binning: AngularBinning::Flat2D {
                min: -0.01,
                max: 0.01,
                bins,
            },
            sigma_ang_mrad: 0.0,
            bg_rate_hz: 0.0,
        }
    }

    fn grid() -> Grid2 {
        // 8 m × 8 m at 0.5 m voxels, centred on the origin.
        Grid2::new([16, 16], [-4.0, -4.0], 0.5)
    }

    #[test]
    fn a_vertical_ray_through_a_uniform_grid_covers_the_scene_height() {
        let scene = Scene2D::new(grid(), vec![detector(0.25, -9.0, 1)]);
        let system = build_system_2d(&scene);
        let rho = vec![1.0f32; system.voxel_count()];
        let opacity = system.opacity(&rho);
        // 8 m = 800 cm of material.
        assert!(
            (opacity[0] - 800.0).abs() < 1e-2,
            "opacity = {}",
            opacity[0]
        );
    }

    #[test]
    fn acceptance_includes_area_efficiency_and_the_zenith_cosine() {
        // Hand-computed bin: area 1 m² = 1e4 cm², efficiency 1, φ = 0 so
        // cos θ = 1, ΔΩ = Δφ = 0.02 rad.
        let scene = Scene2D::new(grid(), vec![detector(0.25, -9.0, 1)]);
        let system = build_system_2d(&scene);
        let expected = 1.0 * 1.0 * 1.0e4 * 1.0 * 0.02;
        assert!(
            (system.rays[0].k_scale - expected).abs() < 1e-2,
            "k_scale = {} expected {expected}",
            system.rays[0].k_scale
        );
    }

    #[test]
    fn the_operator_satisfies_the_adjoint_identity() {
        let scene = Scene2D::new(grid(), vec![detector(-9.0, 0.25, 8)]);
        let system = build_system_2d(&scene);
        let x = vec![1.5f32; system.voxel_count()];
        let y = vec![0.75f32; system.ray_count()];
        let (lhs, rhs) = dot_product_test(&system, &x, &y);
        assert!(
            ((lhs - rhs) / lhs.abs().max(1.0)).abs() < 1e-6,
            "lhs={lhs} rhs={rhs}"
        );
    }

    #[test]
    fn assembly_is_byte_identical_for_the_same_scene() {
        let scene = Scene2D::new(grid(), vec![detector(-9.0, 0.25, 12)]);
        let a = build_system_2d(&scene);
        let b = build_system_2d(&scene);
        assert_eq!(a.l.indptr, b.l.indptr);
        assert_eq!(a.l.indices, b.l.indices);
        assert_eq!(a.l.data, b.l.data);
        assert_eq!(a.rays, b.rays);
    }

    #[test]
    fn a_box_aligned_to_the_grid_reproduces_its_analytic_chord() {
        // The box's faces sit on voxel boundaries (multiples of 0.5 m), so the
        // voxelised density is the box itself and the only remaining error is
        // DDA's own.
        let scene = Scene2D::new(grid(), vec![detector(0.25, -9.0, 1)]);
        let system = build_system_2d(&scene);
        let mut rho = vec![0.0f32; system.voxel_count()];
        let grid = grid();
        for j in 0..grid.n[1] {
            for i in 0..grid.n[0] {
                let [x, y] = grid.centre(i, j);
                if (-2.0..=2.0).contains(&x) && (-1.0..=1.0).contains(&y) {
                    rho[grid.index(i, j).unwrap()] = 1.0;
                }
            }
        }
        let analytic = crate::chords::chord_len(
            &Primitive::Box {
                c: [0.0, 0.0, 0.0],
                half: [2.0, 1.0, 1.0],
                rot: Quat::IDENTITY,
            },
            [0.25, -9.0],
            [0.0, 1.0],
        );
        let opacity = system.opacity(&rho);
        let relative = ((opacity[0] - analytic * 100.0) / (analytic * 100.0)).abs();
        assert!(
            relative < 1e-4,
            "opacity {} vs analytic {} cm (relative {relative})",
            opacity[0],
            analytic * 100.0
        );
    }

    #[test]
    fn the_system_carries_the_pose_every_ray_names() {
        let scene = Scene2D::new(
            grid(),
            vec![detector(0.25, -9.0, 2), detector(-9.0, 0.25, 3)],
        );
        let system = build_system_2d(&scene);
        assert_eq!(system.poses.len(), scene.detectors.len());
        assert_eq!(system.ray_count(), 5, "two bins plus three");
        for (index, ray) in system.rays.iter().enumerate() {
            let pose = system
                .detector_pose(ray.det)
                .unwrap_or_else(|| panic!("ray {index} names detector {} with no pose", ray.det));
            assert_eq!(
                pose.position,
                scene.detectors[usize::from(ray.det)].pose.position,
                "ray {index} must sit at the detector it belongs to"
            );
        }
        // `RayCell::det` is a u16 while a system can hold no poses at all: the
        // hand-written fixtures rely on that, and a caller has to be able to ask
        // without getting a panic back.
        assert!(system.detector_pose(9).is_none());
    }

    #[test]
    fn efficiency_zero_produces_rays_with_no_acceptance() {
        let mut low = detector(0.25, -9.0, 4);
        low.eff = 0.0;
        let scene = Scene2D::new(grid(), vec![low]);
        let system = build_system_2d(&scene);
        assert_eq!(system.ray_count(), 4);
        assert!(system.rays.iter().all(|r| r.k_scale == 0.0));
    }
}
