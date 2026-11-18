//! Native tooling for UMBRA.
//!
//! Subcommands:
//!
//! | Command | Purpose |
//! |---|---|
//! | `umbra gen-flux [--out DIR]` | write the versioned flux table asset and its manifest |
//! | `umbra scene [--scene FILE]` | load, validate and report a scene document (spec §8.1) |
//! | `umbra golden [--scene FILE] [--out FILE]` | regenerate the committed golden dataset |
//! | `umbra selftest [--scene FILE]` | load the default scene, evaluate the forward model, check the adjoint |
//! | `umbra bench` | timings for a 128 x 128 system |
//!
//! The table generator and the golden writer are build-time tools, not runtime
//! dependencies: the app loads the committed `.bin` from
//! `web/public/tables/<version>/` (spec §8.5), and every solver reconstructs the
//! committed golden dataset rather than regenerating it.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use umbra_core::hash::sha256_hex;
use umbra_core::json::{self, Json};
use umbra_core::{dot_product_test, Grid2};
use umbra_forward::scene::SceneSpec;
use umbra_forward::{
    build_system_2d, counts_hash, expected, fidelity_id, Expected, System, GOLDEN_SCHEMA,
};
use umbra_physics::simulate::{ExposureProcess, DEFAULT_CHECKPOINTS};
use umbra_physics::{
    de_min_dx, e_min, generate_default, AngularBinning, Detector, Pose, STANDARD_ROCK_A,
    STANDARD_ROCK_B, TABLE_VERSION,
};

/// The shared default scene (spec §8.1, §8.2).
const DEFAULT_SCENE: &str = "web/src/scenes/default.json";
/// The golden dataset every solver reconstructs.
const DEFAULT_GOLDEN: &str = "crates/umbra-recon/tests/golden/default.json";

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
            gen_flux(&flag(&args, "--out").unwrap_or_else(|| PathBuf::from("web/public/tables/v1")))
        }
        Some("scene") => {
            let spec = load_scene(
                &flag(&args, "--scene").unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE)),
            )?;
            report(&spec);
            Ok(())
        }
        Some("golden") => {
            let scene = flag(&args, "--scene").unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE));
            let out = flag(&args, "--out").unwrap_or_else(|| PathBuf::from(DEFAULT_GOLDEN));
            write_golden(&scene, &out)
        }
        Some("bench") => {
            bench();
            Ok(())
        }
        Some("selftest") | None => {
            let scene = flag(&args, "--scene").unwrap_or_else(|| PathBuf::from(DEFAULT_SCENE));
            selftest(&scene)
        }
        Some(other) => {
            eprintln!("unknown subcommand `{other}`");
            eprintln!(
                "usage: umbra <gen-flux [--out DIR] | scene [--scene FILE] | \
                 golden [--scene FILE] [--out FILE] | selftest [--scene FILE] | bench>"
            );
            std::process::exit(2);
        }
    }
}

/// `--name VALUE`, if present.
fn flag(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
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

/// Loads and validates a scene document, naming the file in any error.
fn load_scene(path: &Path) -> Result<SceneSpec, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    Ok(SceneSpec::from_json_str(&text)?)
}

/// Everything downstream of a scene, computed once.
struct Dataset {
    system: System,
    density: Vec<f32>,
    model: Expected,
    counts: Vec<u32>,
}

fn dataset(spec: &SceneSpec) -> Dataset {
    let system = build_system_2d(&spec.to_scene_2d());
    let density = spec.density();
    let table = generate_default();
    let model = expected(
        &system,
        &density,
        &table,
        spec.physics.level,
        spec.exposure_s,
        &spec.forward_params(),
    );
    let process = ExposureProcess::new(
        &model.lambda,
        spec.exposure_s,
        spec.seed,
        spec.scene_key(),
        DEFAULT_CHECKPOINTS,
    );
    let counts = process.counts(f64::from(spec.exposure_s));
    Dataset {
        system,
        density,
        model,
        counts,
    }
}

fn report(spec: &SceneSpec) {
    let data = dataset(spec);
    let (lower, upper) = spec.grid_bounds();

    println!("{} — {}", spec.id, spec.label);
    if spec.schematic {
        println!("  schematic  true: must carry the §8.3 badge wherever it is shown");
    }
    println!(
        "  grid       {} x {} @ {} m over x [{}..{}] m, y [{}..{}] m",
        spec.grid.n[0], spec.grid.n[1], spec.grid.h_m, lower[0], upper[0], lower[1], upper[1]
    );
    println!(
        "  geometry   {} solid(s), {} void(s), bulk density {} g/cm3",
        spec.primitives.len(),
        spec.voids.len(),
        spec.physics.rho_bulk
    );
    println!(
        "  detectors  {} detector(s), {} ray cells, {} non-zeros",
        spec.detectors.len(),
        data.system.ray_count(),
        data.system.l.nnz()
    );
    println!(
        "  exposure   {} s at {} (seed {})",
        spec.exposure_s,
        fidelity_id(spec.physics.level),
        spec.seed
    );
    println!("  scene key  {}", spec.scene_key_hex());
    println!(
        "  density    min {:.3}  mean {:.3}  max {:.3} g/cm3",
        minimum(&data.density),
        mean(&data.density),
        maximum(&data.density)
    );
    let mut lambda = data.model.lambda.clone();
    lambda.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "  lambda     min {:.1}  median {:.1}  max {:.1} counts",
        lambda.first().copied().unwrap_or(0.0),
        lambda.get(lambda.len() / 2).copied().unwrap_or(0.0),
        lambda.last().copied().unwrap_or(0.0)
    );
    println!(
        "  counts     min {}  max {}  hash {}",
        data.counts.iter().copied().min().unwrap_or(0),
        data.counts.iter().copied().max().unwrap_or(0),
        counts_hash(&data.counts)
    );
}

fn minimum(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn maximum(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// The golden document for a scene, and its count hash.
///
/// `scene` is the path the dataset was generated from, not a constant: once a
/// second scene exists, a golden that names the wrong file is provenance that
/// lies, and nothing downstream would notice.
fn golden_document(spec: &SceneSpec, scene: &Path) -> (Json, String) {
    let data = dataset(spec);
    let hash = counts_hash(&data.counts);
    let scene_path = scene.display().to_string();
    let command = format!("cargo run -p umbra-cli -- golden --scene {scene_path}");
    let document = Json::Object(vec![
        ("schema".to_string(), Json::from(GOLDEN_SCHEMA)),
        ("scene_id".to_string(), Json::from(spec.id.as_str())),
        ("scene_key".to_string(), Json::from(spec.scene_key_hex())),
        ("seed".to_string(), Json::from(spec.seed)),
        ("exposure_s".to_string(), Json::from(spec.exposure_s)),
        (
            "fidelity".to_string(),
            Json::from(fidelity_id(spec.physics.level)),
        ),
        (
            "grid".to_string(),
            Json::Object(vec![
                (
                    "n".to_string(),
                    Json::Array(spec.grid.n.iter().map(|value| Json::from(*value)).collect()),
                ),
                (
                    "origin_m".to_string(),
                    Json::Array(
                        spec.grid
                            .origin
                            .iter()
                            .map(|value| Json::from(*value))
                            .collect(),
                    ),
                ),
                ("h_m".to_string(), Json::from(spec.grid.h_m)),
            ]),
        ),
        (
            "ray_count".to_string(),
            Json::from(data.system.ray_count() as u32),
        ),
        (
            "density".to_string(),
            Json::Array(
                data.density
                    .iter()
                    .map(|value| Json::from(*value))
                    .collect(),
            ),
        ),
        (
            "lambda".to_string(),
            Json::Array(
                data.model
                    .lambda
                    .iter()
                    .map(|value| Json::from(*value))
                    .collect(),
            ),
        ),
        (
            "counts".to_string(),
            Json::Array(data.counts.iter().map(|count| Json::from(*count)).collect()),
        ),
        ("count_hash".to_string(), Json::from(hash.as_str())),
        (
            "generated_by".to_string(),
            Json::Object(vec![
                ("command".to_string(), Json::from(command.as_str())),
                ("scene".to_string(), Json::from(scene_path.as_str())),
                (
                    "umbra_version".to_string(),
                    Json::from(env!("CARGO_PKG_VERSION")),
                ),
            ]),
        ),
    ]);
    (document, hash)
}

fn write_golden(scene: &Path, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let spec = load_scene(scene)?;
    let (document, hash) = golden_document(&spec, scene);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, format!("{}\n", json::to_string_pretty(&document)))?;
    println!(
        "wrote {} (scene {}, count hash {})",
        out.display(),
        spec.id,
        hash
    );
    Ok(())
}

fn selftest(scene: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("umbra {}", env!("CARGO_PKG_VERSION"));
    let table = generate_default();
    println!(
        "flux table: {} nodes, {} bytes serialised",
        table.len(),
        table.to_bytes().len()
    );

    let spec = load_scene(scene)?;
    report(&spec);
    let data = dataset(&spec);

    let opacity = data.model.opacity[0];
    println!(
        "L: opacity {:.0} g/cm2 -> E_min {:.3} GeV -> lambda {:.1} counts (dλ/dX {:.3})",
        opacity,
        e_min(opacity, STANDARD_ROCK_A, STANDARD_ROCK_B),
        data.model.lambda[0],
        data.model.dlambda_dx[0]
    );
    println!(
        "dE_min/dX at that opacity: {:.6} GeV per g/cm2",
        de_min_dx(opacity, STANDARD_ROCK_A, STANDARD_ROCK_B)
    );

    let x = vec![1.0f32; data.system.voxel_count()];
    let y = vec![1.0f32; data.system.ray_count()];
    let (lhs, rhs) = dot_product_test(&data.system, &x, &y);
    println!("adjoint check: <Lx,y> = {lhs:.6}, <x,L^T y> = {rhs:.6}");
    assert!(
        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
        "adjoint identity violated"
    );
    Ok(())
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
    let scene = umbra_forward::Scene2D::new(grid, vec![detector]);

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
        umbra_physics::Fidelity::L1,
        30.0 * 86_400.0,
        &umbra_forward::ForwardParams::default(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Repo-relative paths do not resolve under `cargo test`, which runs with the
    /// crate directory as the working directory.
    fn committed(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative)
    }

    #[test]
    fn the_committed_default_scene_loads_and_has_the_expected_shape() {
        let spec = load_scene(&committed(DEFAULT_SCENE)).expect("the default scene must load");
        assert_eq!(spec.id, "default-slab");
        let system = build_system_2d(&spec.to_scene_2d());
        assert_eq!(system.ray_count(), 32);
        assert_eq!(system.voxel_count(), 1024);
        assert!(system.l.nnz() > 0);
    }

    #[test]
    fn the_default_scene_satisfies_the_adjoint_identity() {
        let spec = load_scene(&committed(DEFAULT_SCENE)).expect("the default scene must load");
        let system = build_system_2d(&spec.to_scene_2d());
        use umbra_core::LinearOperator;
        let x = vec![1.0f32; system.voxel_count()];
        let y = vec![0.5f32; system.ray_count()];
        let (lhs, rhs) = dot_product_test(&system, &x, &y);
        assert!(
            (lhs - rhs).abs() < 1e-5 * lhs.abs().max(1.0),
            "lhs={lhs} rhs={rhs}"
        );
        assert_eq!(LinearOperator::ncols(&system), 1024);
    }

    /// The four-pose scene: several views, and a void no single view can place.
    const MULTIVIEW_SCENE: &str = "web/src/scenes/multiview.json";
    /// The four-pose scene's golden dataset.
    const MULTIVIEW_GOLDEN: &str = "crates/umbra-recon/tests/golden/multiview.json";

    fn assert_golden_is_fresh(scene: &str, golden: &str) {
        // Byte equality of the regenerated document, not a loose comparison: the
        // golden is a reviewed artefact, and a stale one would let a physics
        // change land without anyone reading it (§13.5).
        let path = committed(scene);
        let spec = load_scene(&path).expect("the scene must load");
        // Provenance records the repository-relative path the documented command
        // uses, not the absolute path this test loads from: a golden that named
        // its author's home directory would differ on every machine.
        let (fresh, _) = golden_document(&spec, Path::new(scene));
        let rendered = format!("{}\n", json::to_string_pretty(&fresh));
        let committed_golden = fs::read_to_string(committed(golden))
            .expect("the golden dataset must be committed at crates/umbra-recon/tests/golden");
        assert_eq!(
            rendered, committed_golden,
            "{golden} is stale: run `cargo run -p umbra-cli -- golden --scene {scene} --out {golden}` and review the diff"
        );
    }

    #[test]
    fn the_committed_golden_matches_a_fresh_evaluation() {
        assert_golden_is_fresh(DEFAULT_SCENE, DEFAULT_GOLDEN);
    }

    #[test]
    fn the_multiview_golden_matches_a_fresh_evaluation() {
        // Two scenes, two goldens: a second fixture nobody regenerates is a
        // fixture that quietly stops describing its scene.
        assert_golden_is_fresh(MULTIVIEW_SCENE, MULTIVIEW_GOLDEN);
    }

    #[test]
    fn gen_flux_writes_a_table_and_a_manifest() {
        let dir = env::temp_dir().join(format!("umbra-cli-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        gen_flux(&dir).expect("generation should succeed");

        let bytes = fs::read(dir.join("flux.bin")).expect("table written");
        let table = umbra_physics::FluxTable::from_bytes(&bytes).expect("table parses");
        assert!(table.is_consistent());
        assert_eq!(table.log10_e_min.len(), 129);

        let manifest = fs::read_to_string(dir.join("manifest.json")).expect("manifest written");
        assert!(manifest.contains("\"asset\": \"tables\""));
        assert!(manifest.contains(&sha256_hex(&bytes)));
        let _ = fs::remove_dir_all(&dir);
    }
}
