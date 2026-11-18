//! Amanatides–Woo DDA traversal in the slice plane (spec §5.3).
//!
//! Equivalent to Siddon's method: the ray is advanced from voxel boundary to
//! voxel boundary and each cell receives its exact segment length, so the sum of
//! a row is the geometric chord through the grid rather than a stair-stepped
//! approximation of it.
//!
//! Two properties the callers rely on:
//!
//! - **Deterministic ordering.** Cells are emitted in traversal order, so the same
//!   scene always assembles byte-identical CSR arrays.
//! - **No zero-length entries.** A cell touched at a corner emits nothing for it,
//!   which keeps `nnz` honest.

use umbra_core::Grid2;

/// One traversed cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellHit {
    /// Flat row-major index into the grid.
    pub index: usize,
    /// Length of the ray inside this cell, in metres.
    pub length_m: f32,
}

/// Traverses `grid` from `origin` along `dir` for at most `max_len_m` metres.
///
/// `dir` need not be normalised. A ray that starts outside the grid is advanced
/// to its entry point first, so a detector sitting outside the scene still
/// integrates the structure correctly.
pub fn traverse_2d(grid: &Grid2, origin: [f32; 2], dir: [f32; 2], max_len_m: f32) -> Vec<CellHit> {
    let norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if !norm.is_finite() || norm <= 0.0 || max_len_m <= 0.0 || grid.h_m <= 0.0 {
        return Vec::new();
    }
    let d = [dir[0] / norm, dir[1] / norm];
    let h = grid.h_m;
    let lo = grid.origin;
    let hi = [
        grid.origin[0] + grid.n[0] as f32 * h,
        grid.origin[1] + grid.n[1] as f32 * h,
    ];

    // Clip the ray to the grid: a straight line meets an axis-aligned rectangle
    // in one interval, so entry and exit are enough.
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for axis in 0..2 {
        if d[axis].abs() < 1e-12 {
            if origin[axis] < lo[axis] || origin[axis] > hi[axis] {
                return Vec::new();
            }
            continue;
        }
        let inv = 1.0 / d[axis];
        let t1 = (lo[axis] - origin[axis]) * inv;
        let t2 = (hi[axis] - origin[axis]) * inv;
        t_enter = t_enter.max(t1.min(t2));
        t_exit = t_exit.min(t1.max(t2));
    }
    // The ray budget is material traversed inside the grid, measured from the
    // entry point: a detector sitting outside the scene should not spend its
    // budget crossing empty air.
    let entry = t_enter.max(0.0);
    let exit = t_exit.min(entry + max_len_m);
    if exit <= entry {
        return Vec::new();
    }

    let start = [origin[0] + d[0] * entry, origin[1] + d[1] * entry];
    let mut ix = ((start[0] - lo[0]) / h).floor() as i64;
    let mut iy = ((start[1] - lo[1]) / h).floor() as i64;
    // The exit point can land exactly on the upper face; keep indices in range.
    ix = ix.clamp(0, grid.n[0] as i64 - 1);
    iy = iy.clamp(0, grid.n[1] as i64 - 1);

    let step = [
        if d[0] > 0.0 { 1 } else { -1 },
        if d[1] > 0.0 { 1 } else { -1 },
    ];
    let infinity = f32::INFINITY;

    // Distance along the ray to the next boundary on each axis, and the spacing
    // between boundaries thereafter.
    let mut t_next = [infinity, infinity];
    let mut t_delta = [infinity, infinity];
    for axis in 0..2 {
        if d[axis].abs() < 1e-12 {
            continue;
        }
        let boundary = if axis == 0 {
            lo[0] + (ix + i64::from(step[0] > 0)) as f32 * h
        } else {
            lo[1] + (iy + i64::from(step[1] > 0)) as f32 * h
        };
        t_next[axis] = (boundary - start[axis]) / d[axis];
        t_delta[axis] = h / d[axis].abs();
    }

    // Work in a parameter measured from the entry point, so the clipped start
    // does not have to be carried through the boundary arithmetic.
    let span = exit - entry;
    let mut hits = Vec::new();
    let mut t = 0.0f32;
    loop {
        let exit = t_next[0].min(t_next[1]).min(span);
        let length = exit - t;
        if length > 1e-9 {
            if let Some(index) = grid.index(ix as u32, iy as u32) {
                hits.push(CellHit {
                    index,
                    length_m: length,
                });
            }
        }
        if exit >= span {
            break;
        }
        if t_next[0] <= t_next[1] {
            ix += step[0];
            t = t_next[0];
            t_next[0] += t_delta[0];
        } else {
            iy += step[1];
            t = t_next[1];
            t_next[1] += t_delta[1];
        }
        if ix < 0 || iy < 0 || ix >= grid.n[0] as i64 || iy >= grid.n[1] as i64 {
            break;
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid2 {
        Grid2::new([8, 8], [-2.0, -2.0], 0.5)
    }

    fn total(hits: &[CellHit]) -> f32 {
        hits.iter().map(|h| h.length_m).sum()
    }

    #[test]
    fn a_ray_through_the_middle_covers_the_grid_width() {
        let hits = traverse_2d(&grid(), [-9.0, 0.25], [1.0, 0.0], 100.0);
        assert!(
            (total(&hits) - 4.0).abs() < 1e-5,
            "covered {}",
            total(&hits)
        );
        // One cell per column.
        assert_eq!(hits.len(), 8);
    }

    #[test]
    fn a_diagonal_ray_covers_its_own_length() {
        // From the lower-left corner to the upper-right: the diagonal of a 4 m
        // square, i.e. 4*sqrt(2).
        let hits = traverse_2d(&grid(), [-2.0, -2.0], [1.0, 1.0], 100.0);
        let expected = 4.0 * 2f32.sqrt();
        assert!(
            (total(&hits) - expected).abs() < 1e-4,
            "covered {} expected {expected}",
            total(&hits)
        );
        // Every cell is distinct and in order: no repeats from boundary jitter.
        let mut indices: Vec<usize> = hits.iter().map(|h| h.index).collect();
        let count = indices.len();
        indices.dedup();
        assert_eq!(indices.len(), count, "traversal revisited a cell");
    }

    #[test]
    fn the_row_sums_to_the_chord_through_the_grid() {
        for angle in [0.1f32, 0.35, 1.0, 2.0] {
            let dir = [angle.cos(), angle.sin()];
            let hits = traverse_2d(&grid(), [-2.0, -2.0], dir, 100.0);
            let chord = crate::chords::chord_len(
                &umbra_core::Primitive::Box {
                    c: [0.0, 0.0, 0.0],
                    half: [2.0, 2.0, 1.0],
                    rot: umbra_core::Quat::IDENTITY,
                },
                [-2.0, -2.0],
                dir,
            );
            assert!(
                (total(&hits) - chord).abs() < 1e-4,
                "angle {angle}: covered {} vs chord {chord}",
                total(&hits)
            );
        }
    }

    #[test]
    fn max_length_truncates_the_traversal() {
        let hits = traverse_2d(&grid(), [-9.0, 0.25], [1.0, 0.0], 1.0);
        assert!(
            (total(&hits) - 1.0).abs() < 1e-5,
            "covered {}",
            total(&hits)
        );
    }

    #[test]
    fn a_ray_that_misses_the_grid_traverses_nothing() {
        assert!(traverse_2d(&grid(), [-9.0, 50.0], [1.0, 0.0], 100.0).is_empty());
    }

    #[test]
    fn traversal_is_deterministic() {
        let a = traverse_2d(&grid(), [-9.0, 0.1], [1.0, 0.2], 100.0);
        let b = traverse_2d(&grid(), [-9.0, 0.1], [1.0, 0.2], 100.0);
        assert_eq!(a, b);
    }

    #[test]
    fn degenerate_inputs_are_empty_rather_than_panicking() {
        assert!(traverse_2d(&grid(), [0.0, 0.0], [0.0, 0.0], 10.0).is_empty());
        assert!(traverse_2d(&grid(), [0.0, 0.0], [1.0, 0.0], 0.0).is_empty());
    }
}
