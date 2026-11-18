//! Compressed sparse row matrices.
//!
//! `L` holds path lengths in centimetres: row `i` is one ray cell, column `j`
//! one voxel. `L x` is opacity (forward), `Lᵀ y` is back projection (adjoint),
//! and both are the inner-loop kernels of every solver in the workspace.

use crate::linear::LinearOperator;

/// A compressed sparse row matrix.
///
/// Row `i` occupies `indices[indptr[i] .. indptr[i + 1]]` with matching entries
/// in `data`. Entries need not be sorted by column, and duplicate columns are
/// permitted; both properties are relied on by the jittered sub-ray assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr<T> {
    /// Row offsets, length `nrows + 1`.
    pub indptr: Vec<u32>,
    /// Column index of each stored entry.
    pub indices: Vec<u32>,
    /// Stored value of each entry.
    pub data: Vec<T>,
}

impl<T> Csr<T> {
    pub fn new(indptr: Vec<u32>, indices: Vec<u32>, data: Vec<T>) -> Self {
        assert_eq!(
            indptr.first().copied().unwrap_or(0),
            0,
            "indptr must start at 0"
        );
        assert_eq!(
            indptr.last().copied().unwrap_or(0) as usize,
            data.len(),
            "indptr must terminate at nnz"
        );
        assert_eq!(indices.len(), data.len(), "indices and data must agree");
        Self {
            indptr,
            indices,
            data,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Upper bound on the number of columns: one past the largest column index.
    ///
    /// Because empty trailing columns leave no trace in the sparsity pattern,
    /// this can under-report. Callers that know their grid size should use it
    /// instead (the system builder always does).
    pub fn ncols_hint(&self) -> usize {
        self.indices
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0)
    }

    /// Iterator over `(column, value)` pairs of row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a valid row.
    pub fn row(&self, i: usize) -> Row<'_, T>
    where
        T: Copy,
    {
        assert!(i < self.nrows(), "row {i} out of range");
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        Row {
            indices: &self.indices[start..end],
            data: &self.data[start..end],
        }
    }
}

/// Iterator over the entries of a single CSR row.
pub struct Row<'a, T: Copy> {
    indices: &'a [u32],
    data: &'a [T],
}

impl<T: Copy> Iterator for Row<'_, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        let (index, value) = (*self.indices.first()?, *self.data.first()?);
        self.indices = &self.indices[1..];
        self.data = &self.data[1..];
        Some((index as usize, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len(), Some(self.data.len()))
    }
}

impl<T: Copy> ExactSizeIterator for Row<'_, T> {}

impl Csr<f32> {
    /// `y = L x`, overwriting `y`.
    ///
    /// # Panics
    ///
    /// Panics if `y` is not exactly `nrows` long, or if `x` is shorter than a
    /// column index present in the matrix.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(y.len(), self.nrows(), "y must have one entry per row");
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (j, v) in self.row(i) {
                acc += v * x[j];
            }
            *yi = acc;
        }
    }

    /// `x = Lᵀ y`, overwriting `x`. Each row of `L` scatters into `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is shorter than a column index present in the matrix.
    pub fn matvec_t(&self, y: &[f32], x: &mut [f32]) {
        assert_eq!(y.len(), self.nrows(), "y must have one entry per row");
        x.fill(0.0);
        for (i, yi) in y.iter().enumerate() {
            for (j, v) in self.row(i) {
                x[j] += v * yi;
            }
        }
    }
}

impl LinearOperator for Csr<f32> {
    fn nrows(&self) -> usize {
        Csr::nrows(self)
    }

    fn ncols(&self) -> usize {
        self.ncols_hint()
    }

    fn apply(&self, x: &[f32], y: &mut [f32]) {
        self.matvec(x, y);
    }

    fn apply_t(&self, y: &[f32], x: &mut [f32]) {
        self.matvec_t(y, x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::dot_product_test;

    /// 3x2 matrix:
    /// ```text
    /// [ 1  0 ]
    /// [ 0  2 ]
    /// [ 3  4 ]
    /// ```
    fn sample() -> Csr<f32> {
        Csr::new(vec![0, 1, 2, 4], vec![0, 1, 0, 1], vec![1.0, 2.0, 3.0, 4.0])
    }

    #[test]
    fn shape_and_nnz() {
        let a = sample();
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.nnz(), 4);
        assert_eq!(a.ncols_hint(), 2);
    }

    #[test]
    fn matvec_and_adjoint_agree_with_dense_product() {
        let a = sample();
        let x = [2.0f32, -1.0];
        let mut y = [0.0f32; 3];
        a.matvec(&x, &mut y);
        assert_eq!(y, [2.0, -2.0, 2.0]);

        let w = [1.0f32, 0.5, -2.0];
        let mut x_out = [0.0f32; 2];
        a.matvec_t(&w, &mut x_out);
        // column 0: 1*1 + 3*(-2) = -5 ; column 1: 2*0.5 + 4*(-2) = -7
        assert_eq!(x_out, [-5.0, -7.0]);
    }

    #[test]
    fn adjoint_identity_holds() {
        let a = sample();
        let (lhs, rhs) = dot_product_test(&a, &[0.5, -1.5], &[1.0, 2.0, -0.25]);
        let scale = lhs.abs().max(1.0);
        assert!(((lhs - rhs) / scale).abs() < 1e-5, "lhs={lhs} rhs={rhs}");
    }

    #[test]
    fn rows_iterate_in_stored_order() {
        let a = sample();
        let row: Vec<_> = a.row(2).collect();
        assert_eq!(row, vec![(0, 3.0), (1, 4.0)]);
    }

    #[test]
    #[should_panic(expected = "indptr must terminate at nnz")]
    fn inconsistent_indptr_is_rejected() {
        let _ = Csr::new(vec![0, 2], vec![0], vec![1.0f32]);
    }
}
