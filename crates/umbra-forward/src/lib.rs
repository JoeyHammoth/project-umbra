//! The forward model: scene geometry → ray cells → opacity.
//!
//! A [`System`] is the discretised experiment: one CSR row per ray cell holding
//! path lengths through each voxel, plus the per-ray metadata the physics needs.
//! `L ρ` gives opacity, `Lᵀ ·` back projects, and both are shared with every
//! solver and statistic in the workspace.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`system`] | the CSR operator plus per-ray metadata |
//! | [`dda`] | slice-plane traversal producing exact path lengths |
//! | [`build`] | detectors + bins → rays → assembled system |
//! | [`chords`] | analytic chords, the independent geometry oracle |
//! | [`expected`] | L0/L1 expected counts and their exact gradients |

pub mod build;
pub mod chords;
pub mod dda;
pub mod expected;
pub mod system;

pub use build::{build_system_2d, Scene2D};
pub use chords::chord_len;
pub use dda::{traverse_2d, CellHit};
pub use expected::{expected, Expected, ForwardParams};
pub use system::{RayCell, System};
