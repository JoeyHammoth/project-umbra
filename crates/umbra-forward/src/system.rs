//! The discretised experiment.

use umbra_core::{Csr, GridAny, LinearOperator};

/// One ray cell: a detector, a direction and the acceptance factor that scales
/// its expected counts.
///
/// `k_scale` is `K_i / T` from spec §2.3, i.e. `ε_d A_d cos θ_i ΔΩ_i` — the
/// exposure-independent part of the acceptance. Exposure multiplies it at
/// evaluation time, so time scrubbing never rebuilds the system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCell {
    /// Index of the detector this cell belongs to.
    pub det: u16,
    /// Zenith angle, in radians.
    pub zen: f32,
    /// Azimuth angle, in radians.
    pub az: f32,
    /// `ε_d A_d cos θ ΔΩ`, with area in cm².
    pub k_scale: f32,
}

/// The path-length operator and the ray metadata that indexes it.
#[derive(Debug, Clone, PartialEq)]
pub struct System {
    /// Path lengths in centimetres, one row per ray cell.
    pub l: Csr<f32>,
    /// One entry per row of `l`.
    pub rays: Vec<RayCell>,
    /// The voxel grid the columns refer to.
    pub grid: GridAny,
}

impl LinearOperator for System {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }

    fn ncols(&self) -> usize {
        // The grid size, not the largest column index: trailing voxels that no
        // ray touches are still unknowns in the inverse problem.
        self.grid.cell_count()
    }

    fn apply(&self, x: &[f32], y: &mut [f32]) {
        self.l.matvec(x, y);
    }

    fn apply_t(&self, y: &[f32], x: &mut [f32]) {
        self.l.matvec_t(y, x);
    }
}

impl System {
    /// Number of ray cells (rows of `L`).
    pub fn ray_count(&self) -> usize {
        self.rays.len()
    }

    /// Number of voxels (columns of `L`).
    pub fn voxel_count(&self) -> usize {
        self.grid.cell_count()
    }

    /// Opacity `X = L ρ`, in g cm⁻², for a density field in g cm⁻³.
    ///
    /// # Panics
    ///
    /// Panics if `rho` has fewer entries than the grid has voxels.
    pub fn opacity(&self, rho: &[f32]) -> Vec<f32> {
        assert!(
            rho.len() >= self.voxel_count(),
            "rho must cover every voxel: {} < {}",
            rho.len(),
            self.voxel_count()
        );
        let mut out = vec![0.0f32; self.l.nrows()];
        self.l.matvec(rho, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::Grid2;

    #[test]
    fn opacity_is_the_path_length_product() {
        // Two rays through three voxels.
        let grid = Grid2::new([3, 1], [0.0, 0.0], 1.0);
        let l = Csr::new(
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![100.0, 50.0, 50.0, 100.0],
        );
        let system = System {
            l,
            rays: vec![
                RayCell {
                    det: 0,
                    zen: 0.0,
                    az: 0.0,
                    k_scale: 1.0,
                },
                RayCell {
                    det: 0,
                    zen: 0.1,
                    az: 0.0,
                    k_scale: 0.5,
                },
            ],
            grid: GridAny::Two(grid),
        };
        let rho = [2.0f32, 2.0, 2.0];
        assert_eq!(system.opacity(&rho), vec![300.0, 300.0]);
        assert_eq!(system.ray_count(), 2);
        assert_eq!(system.voxel_count(), 3);
    }
}
