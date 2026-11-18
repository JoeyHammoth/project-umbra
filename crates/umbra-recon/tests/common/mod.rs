//! Shared loader for the golden default-scene dataset.
//!
//! Spec §11.1 makes "a result on the default scene" part of every solver's exit
//! criteria, so every solver's test must reconstruct the *same* data. That is
//! what this module hands out: the committed scene, the density field derived
//! from it, the expected counts `λ`, and the sampled counts a solver actually
//! inverts — loaded from the committed golden file rather than regenerated, so a
//! solver change cannot quietly move its own target.
//!
//! Regenerating the golden is `cargo run -p umbra-cli -- golden`; the CLI test
//! `the_committed_golden_matches_a_fresh_evaluation` fails if it is stale.

use std::path::{Path, PathBuf};

use umbra_core::json::{self, Json};
use umbra_core::Grid2;
use umbra_forward::scene::SceneSpec;
use umbra_forward::{build_system_2d, System};

/// The shared default scene, relative to the repository root.
pub const SCENE_PATH: &str = "web/src/scenes/default.json";
/// The golden dataset, relative to the repository root.
pub const GOLDEN_PATH: &str = "crates/umbra-recon/tests/golden/default.json";

/// A loaded golden dataset: everything a solver needs, and nothing it does not.
pub struct Golden {
    /// The scene the dataset came from.
    pub scene: SceneSpec,
    /// Content key of that scene, as recorded in the golden file.
    pub scene_key: String,
    /// RNG seed for the exposure simulator.
    pub seed: u64,
    /// Exposure, in seconds.
    pub exposure_s: f32,
    /// Density field, one entry per voxel, in g cm⁻³.
    pub density: Vec<f32>,
    /// Expected counts `λ`, one entry per ray cell.
    pub lambda: Vec<f32>,
    /// Sampled counts, one entry per ray cell.
    pub counts: Vec<u32>,
    /// SHA-256 of `counts`.
    pub count_hash: String,
}

/// Path to a repository-relative file.
pub fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn read(relative: &str) -> String {
    let path = repo_path(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn number(value: &Json, key: &str) -> f64 {
    value
        .get(key)
        .and_then(Json::as_f64)
        .unwrap_or_else(|| panic!("golden is missing numeric `{key}`: {value:?}"))
}

fn numbers(value: &Json, key: &str) -> Vec<f64> {
    value
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("golden is missing array `{key}`"))
        .iter()
        .map(|item| {
            item.as_f64()
                .unwrap_or_else(|| panic!("golden `{key}` holds a non-number"))
        })
        .collect()
}

fn text<'a>(value: &'a Json, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("golden is missing string `{key}`"))
}

/// Loads the scene and the golden dataset it was generated from.
pub fn load() -> Golden {
    let scene = SceneSpec::from_json_str(&read(SCENE_PATH))
        .unwrap_or_else(|error| panic!("{SCENE_PATH}: {error}"));

    let golden_text = read(GOLDEN_PATH);
    let golden = json::parse(&golden_text).unwrap_or_else(|error| panic!("{GOLDEN_PATH}: {error}"));
    assert_eq!(
        text(&golden, "schema"),
        umbra_forward::GOLDEN_SCHEMA,
        "{GOLDEN_PATH} was written by a different generator"
    ); // The golden restates the grid so a dataset can be read without the scene;
       // it must agree with the scene, or the two were regenerated apart.
    let grid_json = golden.get("grid").expect("golden carries its grid");
    let n = numbers(grid_json, "n");
    let origin = numbers(grid_json, "origin_m");
    let golden_grid = Grid2::new(
        [n[0] as u32, n[1] as u32],
        [origin[0] as f32, origin[1] as f32],
        number(grid_json, "h_m") as f32,
    );
    assert_eq!(
        golden_grid, scene.grid,
        "{GOLDEN_PATH} was generated from a different grid than {SCENE_PATH} declares"
    );

    Golden {
        scene,
        scene_key: text(&golden, "scene_key").to_string(),
        seed: number(&golden, "seed") as u64,
        exposure_s: number(&golden, "exposure_s") as f32,
        density: numbers(&golden, "density")
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        lambda: numbers(&golden, "lambda")
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        counts: numbers(&golden, "counts")
            .into_iter()
            .map(|value| value as u32)
            .collect(),
        count_hash: text(&golden, "count_hash").to_string(),
    }
}

/// The operator, density field and counts a solver inverts, in one call.
pub fn system_and_counts() -> (System, Vec<f32>, Vec<u32>) {
    let golden = load();
    let system = build_system_2d(&golden.scene.to_scene_2d());
    (system, golden.density, golden.counts)
}
