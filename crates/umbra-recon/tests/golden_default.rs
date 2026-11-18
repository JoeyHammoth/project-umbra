//! The golden dataset every solver reconstructs (spec §11.1).
//!
//! This test is the contract behind M2's exit criterion "a result on the default
//! scene": it proves the dataset loads, that it is the dataset the committed
//! scene describes, and that the counts in it are reproducible from
//! `(scene, seed, exposure)`. A solver test that uses [`common::system_and_counts`]
//! therefore inverts data that is pinned here, not data it invented.

mod common;

use umbra_forward::{build_system_2d, counts_hash, expected};
use umbra_physics::generate_default;
use umbra_physics::simulate::{ExposureProcess, DEFAULT_CHECKPOINTS};

#[test]
fn the_golden_dataset_reproduces_from_the_committed_scene() {
    let golden = common::load();

    assert_eq!(
        golden.scene.scene_key_hex(),
        golden.scene_key,
        "the scene and the golden disagree about which scene this is: one was \
         regenerated without the other"
    );
    assert_eq!(golden.scene.seed, golden.seed);
    assert_eq!(golden.scene.exposure_s, golden.exposure_s);

    let system = build_system_2d(&golden.scene.to_scene_2d());
    assert_eq!(
        system.ray_count(),
        golden.counts.len(),
        "the golden counts do not match the scene's ray count"
    );
    assert_eq!(
        system.voxel_count(),
        golden.density.len(),
        "the golden density does not match the scene's voxel count"
    );

    // The density is derived from the scene, not a second copy of it.
    for (index, (derived, stored)) in golden
        .scene
        .density()
        .iter()
        .zip(&golden.density)
        .enumerate()
    {
        assert!(
            (derived - stored).abs() <= 1e-6,
            "voxel {index}: scene says {derived} g/cm3, golden says {stored}"
        );
    }

    // λ reproduces from the density, so a physics change cannot hide behind a
    // committed count vector.
    let table = generate_default();
    let model = expected(
        &system,
        &golden.density,
        &table,
        golden.scene.physics.level,
        golden.exposure_s,
        &golden.scene.forward_params(),
    );
    for (index, (fresh, stored)) in model.lambda.iter().zip(&golden.lambda).enumerate() {
        let relative = (fresh - stored).abs() / stored.abs().max(1e-6);
        assert!(
            relative <= 1e-5,
            "ray {index}: lambda {fresh} vs golden {stored} (relative {relative:.2e})"
        );
    }

    // And the counts reproduce from (scene key, seed, exposure) alone.
    let process = ExposureProcess::new(
        &model.lambda,
        golden.exposure_s,
        golden.seed,
        golden.scene.scene_key(),
        DEFAULT_CHECKPOINTS,
    );
    let counts = process.counts(f64::from(golden.exposure_s));
    assert_eq!(
        counts, golden.counts,
        "sampled counts drifted from the golden"
    );
    assert_eq!(counts_hash(&counts), golden.count_hash);
}

#[test]
fn the_dataset_a_solver_receives_is_identical_on_every_load() {
    // Two loads, and the convenience accessor, must agree: a solver comparison
    // between A1 and A2 is worthless if their fixtures differ by a float.
    let first = common::load();
    let second = common::load();
    assert_eq!(first.scene_key, second.scene_key);
    assert_eq!(first.density, second.density);
    assert_eq!(first.lambda, second.lambda);
    assert_eq!(first.counts, second.counts);

    let (system, density, counts) = common::system_and_counts();
    assert_eq!(system.ray_count(), counts.len());
    assert_eq!(system.voxel_count(), density.len());
    assert_eq!(counts, first.counts);
    assert_eq!(density, first.density);
}
