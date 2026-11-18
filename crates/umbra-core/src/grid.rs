//! Regular voxel grids.
//!
//! Voxels are cubic and axis-aligned, in metres. Indexing is row-major with the
//! first axis varying fastest, which matches the CSR row layout the system
//! builder produces and keeps `L` row traversal cache-friendly.

use std::fmt;

/// A regular 2-D grid of square cells.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid2 {
    /// Number of cells along each axis.
    pub n: [u32; 2],
    /// World coordinate of the lower corner of cell `(0, 0)`, in metres.
    pub origin: [f32; 2],
    /// Cell size, in metres.
    pub h_m: f32,
}

impl Grid2 {
    pub fn new(n: [u32; 2], origin: [f32; 2], h_m: f32) -> Self {
        Self { n, origin, h_m }
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> usize {
        self.n[0] as usize * self.n[1] as usize
    }

    /// Flat index of cell `(i, j)`, or `None` when outside the grid.
    pub fn index(&self, i: u32, j: u32) -> Option<usize> {
        if i < self.n[0] && j < self.n[1] {
            Some(j as usize * self.n[0] as usize + i as usize)
        } else {
            None
        }
    }

    /// World coordinate of the centre of cell `(i, j)`, in metres.
    pub fn centre(&self, i: u32, j: u32) -> [f32; 2] {
        [
            self.origin[0] + (i as f32 + 0.5) * self.h_m,
            self.origin[1] + (j as f32 + 0.5) * self.h_m,
        ]
    }
}

/// A regular 3-D grid of cubic voxels.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid3 {
    /// Number of voxels along each axis.
    pub n: [u32; 3],
    /// World coordinate of the lower corner of voxel `(0, 0, 0)`, in metres.
    pub origin: [f32; 3],
    /// Voxel size, in metres.
    pub h_m: f32,
}

impl Grid3 {
    pub fn new(n: [u32; 3], origin: [f32; 3], h_m: f32) -> Self {
        Self { n, origin, h_m }
    }

    /// Total number of voxels.
    pub fn cell_count(&self) -> usize {
        self.n[0] as usize * self.n[1] as usize * self.n[2] as usize
    }

    /// Flat index of voxel `(i, j, k)`, or `None` when outside the grid.
    pub fn index(&self, i: u32, j: u32, k: u32) -> Option<usize> {
        if i < self.n[0] && j < self.n[1] && k < self.n[2] {
            let (ni, nj) = (self.n[0] as usize, self.n[1] as usize);
            Some(k as usize * ni * nj + j as usize * ni + i as usize)
        } else {
            None
        }
    }

    /// World coordinate of the centre of voxel `(i, j, k)`, in metres.
    pub fn centre(&self, i: u32, j: u32, k: u32) -> [f32; 3] {
        [
            self.origin[0] + (i as f32 + 0.5) * self.h_m,
            self.origin[1] + (j as f32 + 0.5) * self.h_m,
            self.origin[2] + (k as f32 + 0.5) * self.h_m,
        ]
    }
}

/// Either grid, so a [`crate::Csr`] system can be built for 2-D or 3-D scenes.
#[derive(Debug, Clone, PartialEq)]
pub enum GridAny {
    Two(Grid2),
    Three(Grid3),
}

impl GridAny {
    /// Total number of cells or voxels.
    pub fn cell_count(&self) -> usize {
        match self {
            GridAny::Two(g) => g.cell_count(),
            GridAny::Three(g) => g.cell_count(),
        }
    }

    /// Dimensionality, 2 or 3.
    pub fn dim(&self) -> u8 {
        match self {
            GridAny::Two(_) => 2,
            GridAny::Three(_) => 3,
        }
    }

    /// Voxel size, in metres.
    pub fn h_m(&self) -> f32 {
        match self {
            GridAny::Two(g) => g.h_m,
            GridAny::Three(g) => g.h_m,
        }
    }
}

impl fmt::Display for GridAny {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridAny::Two(g) => write!(f, "Grid2 {:?} @ {} m", g.n, g.h_m),
            GridAny::Three(g) => write!(f, "Grid3 {:?} @ {} m", g.n, g.h_m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_index_is_row_major() {
        let g = Grid2::new([4, 3], [-1.0, -1.0], 0.5);
        assert_eq!(g.cell_count(), 12);
        assert_eq!(g.index(0, 0), Some(0));
        assert_eq!(g.index(3, 0), Some(3));
        assert_eq!(g.index(0, 1), Some(4));
        assert_eq!(g.index(4, 0), None);
        assert_eq!(g.index(0, 3), None);
    }

    #[test]
    fn centres_are_offset_by_half_a_cell() {
        let g = Grid3::new([2, 2, 2], [0.0, 0.0, 0.0], 1.0);
        assert_eq!(g.centre(0, 0, 0), [0.5, 0.5, 0.5]);
        assert_eq!(g.centre(1, 0, 0), [1.5, 0.5, 0.5]);
        assert_eq!(g.cell_count(), 8);
        assert_eq!(g.index(1, 1, 1), Some(7));
    }

    #[test]
    fn grid_any_reports_shape() {
        let g = GridAny::Three(Grid3::new([8, 8, 8], [0.0; 3], 0.25));
        assert_eq!(g.dim(), 3);
        assert_eq!(g.cell_count(), 512);
        assert_eq!(g.h_m(), 0.25);
    }
}
