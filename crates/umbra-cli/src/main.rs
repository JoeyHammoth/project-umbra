//! Native tooling for UMBRA.
//!
//! Subcommands:
//!
//! | Command | Purpose |
//! |---|---|
//! | `umbra gen-flux [--out DIR]` | write the versioned flux table asset and its manifest |
//! | `umbra selftest` | build a small scene, evaluate the forward model, check the adjoint |
//!
//! The table generator is a build-time tool, not a runtime dependency: the app
//! loads the committed `.bin` from `web/public/tables/<version>/` (spec §8.5).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use umbra_core::hash::sha256_hex;
use umbra_core::{dot_product_test, Grid2};
use umbra_forward::{build_system_2d, expected, ForwardParams, Scene2D};
use umbra_physics::{de_min_dx, e_min, STANDARD_ROCK_A, STANDARD_ROCK_B};
use umbra_physics::{generate_default, AngularBinning, Detector, Fidelity, Pose, TABLE_VERSION};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gen-flux") => {
            let out = output_dir(&args);
            gen_flux(&out)
        }
        Some("bench") => {
            bench();
            Ok(())
        }
        Some("selftest") | None => {
            selftest();
            Ok(())
        }
        Some(other) => {
            eprintln!("unknown subcommand `{other}`");
            eprintln!("usage: umbra <gen-flux [--out DIR] | selftest | bench>");
            std::process::exit(2);
        }
    }
}

/// `--out DIR`, defaulting to the committed asset directory.
fn output_dir(args: &[String]) -> PathBuf {
    args.iter()
        .position(|arg| arg == "--out")
        .and_then(|i| args.get(i + 1))
        .map_or_else(|| PathBuf::from("web/public/tables/v1"), PathBuf::from)
}

fn gen_flux(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    let table = generate_default();
    let bytes = table.to_bytes();
    fs::write(out.join("flux.bin"), &bytes)?;

    let digest = sha256_hex(&bytes);
    let manifest = format!(
        "{{\n  \"schema\": \"umbra.assets/1\",\n  \"asset\": \"tables\",\n  \"version\": {TABLE_VERSION},\n  \"files\": [\n    {{ \"path\": \"flux.bin\", \"bytes\": {}, \"sha256\": \"{digest}\" }}\n  ],\n  \"grid\": {{ \"n_e\": {}, \"n_cos\": {} }},\n  \"citation\": \"Gaisser (1990); Guan et al. (2015) arXiv:1509.06176 Eq. 3\"\n}}\n",
        bytes.len(),
        table.log10_e_min.len(),
        table.cos_theta.len()
    );
    fs::write(out.join("manifest.json"), manifest)?;

    println!(
        "wrote {} ({} bytes, {} x {} grid, sha256 {})",
        out.join("flux.bin").display(),
        bytes.len(),
        table.log10_e_min.len(),
        table.cos_theta.len(),
        &digest[..16]
    );
    Ok(())
}

fn selftest() {
    println!("umbra {}", env!("CARGO_PKG_VERSION"));
    let table = generate_default();
    println!(
        "flux table: {} nodes, {} bytes serialised",
        table.len(),
        table.to_bytes().len()
    );

    let scene = demo_scene();
    let system = build_system_2d(&scene);
    println!(
        "demo system: {} rays, {} voxels, {} non-zeros",
        system.ray_count(),
        system.voxel_count(),
        system.l.nnz()
    );

    let rho = vec![2.65f32; system.voxel_count()];
    let result = expected(
        &system,
        &rho,
        &table,
        Fidelity::L1,
        30.0 * 86_400.0,
        &ForwardParams::default(),
    );
    println!(
        "L1: opacity {:.0} g/cm2 -> E_min {:.3} GeV -> lambda {:.1} counts (dλ/dX {:.3})",
        result.opacity[0],
        e_min(result.opacity[0], STANDARD_ROCK_A, STANDARD_ROCK_B),
        result.lambda[0],
        result.dlambda_dx[0]
    );
    println!(
        "dE_min/dX at that opacity: {:.6} GeV per g/cm2",
        de_min_dx(result.opacity[0], STANDARD_ROCK_A, STANDARD_ROCK_B)
    );

    let x = vec![1.0f32; system.voxel_count()];
    let y = vec![1.0f32; system.ray_count()];
    let (lhs, rhs) = dot_product_test(&system, &x, &y);
    println!("adjoint check: <Lx,y> = {lhs:.6}, <x,L^T y> = {rhs:.6}");
    assert!(
        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
        "adjoint identity violated"
    );
}

/// Times the two numbers the M1 acceptance criteria name: assembling a
/// 128 x 128 system with 360 bins, and evaluating the forward model for it.
fn bench() {
    use std::time::Instant;

    let grid = Grid2::new([128, 128], [-32.0, -32.0], 0.5);
    let detector = Detector {
        pose: Pose {
            position: [0.25, -33.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            up: [1.0, 0.0, 0.0],
        },
        area_m2: 0.25,
        eff: 0.9,
        binning: AngularBinning::Flat2D {
            min: -0.6,
            max: 0.6,
            bins: 360,
        },
        sigma_ang_mrad: 10.0,
        bg_rate_hz: 0.05,
    };
    let scene = Scene2D::new(grid, vec![detector]);

    let start = Instant::now();
    let system = build_system_2d(&scene);
    let build = start.elapsed();

    let table = generate_default();
    let rho = vec![2.65f32; system.voxel_count()];
    let start = Instant::now();
    let result = expected(
        &system,
        &rho,
        &table,
        Fidelity::L1,
        30.0 * 86_400.0,
        &ForwardParams::default(),
    );
    let evaluation = start.elapsed();

    println!(
        "build_system_2d: {build:?} ({} rays, {} voxels, {} nnz)",
        system.ray_count(),
        system.voxel_count(),
        system.l.nnz()
    );
    println!(
        "expected (L1):   {evaluation:?} ({} ray cells, first lambda {:.1})",
        result.lambda.len(),
        result.lambda[0]
    );
}

/// A 16 m square of rock at 0.5 m voxels with one detector underneath.
fn demo_scene() -> Scene2D {
    let grid = Grid2::new([32, 32], [-8.0, -8.0], 0.5);
    let detector = Detector {
        pose: Pose {
            position: [0.25, -9.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            up: [1.0, 0.0, 0.0],
        },
        area_m2: 0.25,
        eff: 0.9,
        binning: AngularBinning::Flat2D {
            min: -0.5,
            max: 0.5,
            bins: 32,
        },
        sigma_ang_mrad: 10.0,
        bg_rate_hz: 0.05,
    };
    Scene2D::new(grid, vec![detector])
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::LinearOperator;
    use umbra_physics::FluxTable;

    #[test]
    fn demo_system_is_square_and_has_rays() {
        let system = build_system_2d(&demo_scene());
        assert_eq!(system.ray_count(), 32);
        assert_eq!(system.voxel_count(), 1024);
        assert!(system.l.nnz() > 0);
    }

    #[test]
    fn demo_system_satisfies_the_adjoint_identity() {
        let system = build_system_2d(&demo_scene());
        let x = vec![1.0f32; system.voxel_count()];
        let y = vec![0.5f32; system.ray_count()];
        let (lhs, rhs) = dot_product_test(&system, &x, &y);
        // Both sides are f32 sums of order 1e4, so the check has to be relative.
        assert!(
            (lhs - rhs).abs() < 1e-5 * lhs.abs().max(1.0),
            "lhs={lhs} rhs={rhs}"
        );
        assert_eq!(LinearOperator::ncols(&system), 1024);
    }

    #[test]
    fn gen_flux_writes_a_table_and_a_manifest() {
        let dir = env::temp_dir().join(format!("umbra-cli-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        gen_flux(&dir).expect("generation should succeed");

        let bytes = fs::read(dir.join("flux.bin")).expect("table written");
        let table = FluxTable::from_bytes(&bytes).expect("table parses");
        assert!(table.is_consistent());
        assert_eq!(table.log10_e_min.len(), 129);

        let manifest = fs::read_to_string(dir.join("manifest.json")).expect("manifest written");
        assert!(manifest.contains("\"asset\": \"tables\""));
        assert!(manifest.contains(&sha256_hex(&bytes)));
        let _ = fs::remove_dir_all(&dir);
    }
}
