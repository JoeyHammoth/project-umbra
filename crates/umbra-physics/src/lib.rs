//! The muon physics that turns a density field into expected counts.
//!
//! Chain of quantities: density field → opacity `X = L ρ` → minimum surface
//! energy `E_min(X)` → integral flux `Φ(>E_min, θ)` → expected counts `λ`.
//!
//! Fidelity levels (spec §2.3):
//!
//! | Level | Physics |
//! |---|---|
//! | L0 | Beer–Lambert toy model, closed-form transmission |
//! | L1 | Gaisser flux with Guan et al. large-angle correction, CSDA range–energy |
//! | L2 | L1 plus Highland multiple-scattering blur and a soft-particle background |
//!
//! The flux parametrisation and its table are in [`flux`]; the exposure
//! simulator is in [`simulate`]. The forward model itself lives in
//! `umbra-forward`, because it needs the ray-cell metadata that crate owns.

pub mod constants;
pub mod detector;
pub mod flux;
pub mod range;
pub mod simulate;

pub use constants::{STANDARD_ROCK_A, STANDARD_ROCK_B, STANDARD_ROCK_RHO, STANDARD_ROCK_X0};
pub use detector::{AngularBinning, Detector, Pose};
pub use flux::{
    cos_theta_star, gaisser_guan_diff, generate_default, integral_flux, FluxTable, TableError,
    TABLE_VERSION,
};
pub use range::{de_min_dx, e_min};

/// Which forward-model fidelity to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fidelity {
    /// Beer–Lambert toy model.
    L0,
    /// Gaisser flux with CSDA energy loss.
    L1,
    /// L1 plus multiple scattering and background.
    L2,
}

impl Fidelity {
    /// The default fidelity used by scenarios.
    pub const DEFAULT: Fidelity = Fidelity::L1;
}
