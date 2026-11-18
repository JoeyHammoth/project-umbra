//! Visibility and resolution analysis (algorithm A10).
//!
//! Answers "what can this detector arrangement *not* see?" with three views: a
//! Fisher-information sensitivity map, a local point-spread function per voxel,
//! and the leading null-space modes. The sensitivity map is implemented here
//! because every other view is normalised against it.

use umbra_core::Csr;

/// Diagonal of the Fisher information matrix, one entry per voxel.
///
/// `F_jj = Σ_i L_ij² w_i` with per-ray weights
/// `w_i = (∂λ_i/∂X_i)² / λ_i`. Inverting the diagonal gives the Cramér–Rao
/// bound on each voxel's variance, which is the cheap error bar shown while a
/// full covariance is unavailable.
///
/// # Panics
///
/// Panics if `weights` does not have one entry per row of `l`.
pub fn fisher_diagonal(l: &Csr<f32>, weights: &[f32]) -> Vec<f32> {
    assert_eq!(
        weights.len(),
        l.nrows(),
        "one weight per ray cell (row of L)"
    );
    let ncols = l.ncols_hint();
    let mut f = vec![0.0f32; ncols];
    for (i, &w) in weights.iter().enumerate() {
        for (j, length) in l.row(i) {
            if j < ncols {
                f[j] += length * length * w;
            }
        }
    }
    f
}

/// One bin of an angular coverage rose: a direction and how many rays pass
/// through a voxel along it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageBin {
    /// Azimuth, in radians.
    pub az: f32,
    /// Zenith, in radians.
    pub zen: f32,
    /// Fraction of the total ray weight in this direction, in `[0, 1]`.
    pub weight: f32,
}

/// The directions through one voxel; empty bins are the missing wedge.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageRose {
    /// The voxel this rose describes.
    pub voxel: usize,
    /// Bins, sorted by descending weight.
    pub bins: Vec<CoverageBin>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_diagonal_squares_the_path_lengths() {
        // Row 0 passes voxel 0 for 100 cm, row 1 passes it for 50 cm;
        // both weigh 0.25.
        let l = Csr::new(vec![0, 1, 2], vec![0, 0], vec![100.0, 50.0]);
        let f = fisher_diagonal(&l, &[0.25, 0.25]);
        assert_eq!(f.len(), 1);
        assert!((f[0] - (10_000.0 + 2_500.0) * 0.25).abs() < 1e-3);
    }

    #[test]
    fn zero_weights_give_zero_information() {
        let l = Csr::new(vec![0, 1], vec![0], vec![42.0]);
        assert_eq!(fisher_diagonal(&l, &[0.0]), vec![0.0]);
    }
}
