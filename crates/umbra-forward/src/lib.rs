//! The forward model: scene geometry → ray cells → opacity.
//!
//! A [`System`] is the discretised experiment: one CSR row per ray cell holding
//! path lengths through each voxel, plus the per-ray metadata the physics needs.
//! `L ρ` gives opacity, `Lᵀ ·` back projects, and both are shared with every
//! solver and statistic in the workspace.

pub mod system;

pub use system::{RayCell, System};
