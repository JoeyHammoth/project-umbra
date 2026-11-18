//! Core data structures shared by every UMBRA crate.
//!
//! This crate owns the vocabulary the rest of the workspace speaks: regular
//! grids, the compressed sparse row operator that encodes path lengths, the
//! `LinearOperator` trait with its adjoint, and the analytic primitives used for
//! scene geometry, void templates and analytic chord models.
//!
//! It has no dependencies, so it compiles unchanged to `wasm32-unknown-unknown`
//! and can be used by native tooling.
//!
//! # Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`grid`] | `Grid2` / `Grid3` / `GridAny` — voxel indexing and centres |
//! | [`csr`] | `Csr<T>` sparse matrix with forward and adjoint products |
//! | [`linear`] | [`LinearOperator`] and the dot-product adjoint test |
//! | [`primitive`] | boxes, ellipsoids, capsules and 2-D prisms |
//! | [`rng`] | counter-based Philox streams and Poisson/binomial sampling |
//! | [`hash`] | SHA-256 for asset manifest integrity |

pub mod csr;
pub mod grid;
pub mod hash;
pub mod linear;
pub mod primitive;
pub mod rng;

pub use csr::Csr;
pub use grid::{Grid2, Grid3, GridAny};
pub use hash::{sha256, sha256_hex};
pub use linear::{dot_product_test, LinearOperator};
pub use primitive::{Primitive, Quat};
pub use rng::{stream_id, Philox};
