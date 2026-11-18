//! Native tooling for UMBRA.
//!
//! The real subcommands (flux-table generation, Python-oracle parity runs,
//! SBI dataset dumps, benchmarks) arrive with their roadmap tickets. For now the
//! binary is a smoke check that the crates link and the adjoint identity holds
//! on a hand-built system, which is exactly what CI needs to guard the workspace
//! boundary.

use umbra_core::{dot_product_test, Csr, Grid2, GridAny};
use umbra_forward::{RayCell, System};
use umbra_physics::{de_min_dx, e_min, STANDARD_ROCK_A, STANDARD_ROCK_B};

fn main() {
    println!("umbra {}", env!("CARGO_PKG_VERSION"));

    let system = demo_system();
    println!(
        "demo system: {} rays, {} voxels, {} non-zeros",
        system.ray_count(),
        system.voxel_count(),
        system.l.nnz()
    );

    let rho = [2.65f32; 9];
    let opacity = system.opacity(&rho);
    println!(
        "opacity of a 3 voxel path at 2.65 g/cm3: {:.1} g/cm2 -> E_min {:.3} GeV, dE_min/dX {:.5}",
        opacity[0],
        e_min(opacity[0], STANDARD_ROCK_A, STANDARD_ROCK_B),
        de_min_dx(opacity[0], STANDARD_ROCK_A, STANDARD_ROCK_B)
    );

    let x = [1.0f32, -2.0, 0.5, 0.0, 3.0, -1.0, 0.25, 1.5, -0.5];
    let y = [0.5f32, -1.0, 2.0];
    let (lhs, rhs) = dot_product_test(&system.l, &x, &y);
    println!("adjoint check: <Lx,y> = {lhs:.6}, <x,L^T y> = {rhs:.6}");
    assert!(
        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
        "adjoint identity violated"
    );
}

/// A three-by-three grid with three horizontal rays, one per row.
fn demo_system() -> System {
    let n = [3u32, 3];
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n[1] {
        for col in 0..n[0] {
            indices.push(row * n[0] + col);
            data.push(100.0f32);
        }
        indptr.push(indices.len() as u32);
    }
    System {
        l: Csr::new(indptr, indices, data),
        rays: (0..n[1])
            .map(|row| RayCell {
                det: 0,
                zen: 0.1 * row as f32,
                az: 0.0,
                k_scale: 1.0,
            })
            .collect(),
        grid: GridAny::Two(Grid2::new(n, [0.0, 0.0], 1.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_system_is_square() {
        let system = demo_system();
        assert_eq!(system.ray_count(), 3);
        assert_eq!(system.voxel_count(), 9);
        assert_eq!(system.l.nnz(), 9);
    }

    #[test]
    fn demo_system_satisfies_the_adjoint_identity() {
        let system = demo_system();
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = [1.0f32, -1.0, 0.5];
        let (lhs, rhs) = dot_product_test(&system.l, &x, &y);
        assert!((lhs - rhs).abs() < 1e-4);
    }
}
