//! Linear operators and their adjoints.
//!
//! Every operator in UMBRA — the path-length matrix `L`, its matrix-free twin,
//! the blur kernel, the finite-difference gradient `∇` — implements
//! [`LinearOperator`]. Solvers are written against the trait, so the same code
//! runs on a CSR matrix and on an operator that recomputes traversal on the fly.

/// A linear operator with an explicit adjoint.
///
/// Implementors must satisfy the adjoint identity `⟨A x, y⟩ = ⟨x, Aᵀ y⟩`; the
/// tests use [`dot_product_test`] to enforce it.
pub trait LinearOperator {
    /// Number of rows (ray cells for `L`).
    fn nrows(&self) -> usize;

    /// Number of columns (voxels for `L`).
    fn ncols(&self) -> usize;

    /// `y = A x`, overwriting `y`.
    fn apply(&self, x: &[f32], y: &mut [f32]);

    /// `x = Aᵀ y` (back projection), overwriting `x`.
    fn apply_t(&self, y: &[f32], x: &mut [f32]);
}

/// Computes both sides of the adjoint identity for supplied probe vectors.
///
/// Returns `(⟨A x, y⟩, ⟨x, Aᵀ y⟩)`, accumulated in `f64` so the comparison is
/// not limited by `f32` summation error. Callers assert the two agree to a
/// relative tolerance.
///
/// # Panics
///
/// Panics if `x` or `y` have the wrong length for `op`.
pub fn dot_product_test<O>(op: &O, x: &[f32], y: &[f32]) -> (f64, f64)
where
    O: LinearOperator + ?Sized,
{
    assert_eq!(x.len(), op.ncols(), "probe x must have ncols entries");
    assert_eq!(y.len(), op.nrows(), "probe y must have nrows entries");

    let mut ax = vec![0.0f32; op.nrows()];
    op.apply(x, &mut ax);

    let mut aty = vec![0.0f32; op.ncols()];
    op.apply_t(y, &mut aty);

    let lhs = ax
        .iter()
        .zip(y)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let rhs = aty
        .iter()
        .zip(x)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();

    (lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deliberately transposed-looking operator: scaling plus shift.
    struct ScaleShift {
        scale: f32,
    }

    impl LinearOperator for ScaleShift {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn apply(&self, x: &[f32], y: &mut [f32]) {
            y[0] = self.scale * x[0];
            y[1] = self.scale * x[1];
            y[2] = self.scale * x[2];
        }
        fn apply_t(&self, y: &[f32], x: &mut [f32]) {
            self.apply(y, x);
        }
    }

    #[test]
    fn dot_product_test_agrees_for_symmetric_operator() {
        let op = ScaleShift { scale: 2.5 };
        let (lhs, rhs) = dot_product_test(&op, &[1.0, -2.0, 0.5], &[3.0, 0.25, -1.0]);
        assert!((lhs - rhs).abs() < 1e-6, "lhs={lhs} rhs={rhs}");
    }
}
