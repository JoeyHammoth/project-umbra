//! Material constants for standard rock.
//!
//! These are the defaults named in spec §2.3. Any deployment can override them
//! per scenario; they are constants rather than a struct because the forward
//! model takes them as plain arguments and the solver code must stay allocation
//! free.

/// Ionisation term `a` in `-dE/dX = a + b E`, in GeV cm² g⁻¹.
pub const STANDARD_ROCK_A: f32 = 2.0e-3;

/// Radiative term `b` in `-dE/dX = a + b E`, in cm² g⁻¹.
pub const STANDARD_ROCK_B: f32 = 4.0e-6;

/// Bulk density of standard rock, in g cm⁻³.
pub const STANDARD_ROCK_RHO: f32 = 2.65;

/// Radiation length of standard rock, in g cm⁻².
pub const STANDARD_ROCK_X0: f32 = 26.5;

/// Critical energy `a / b`, in GeV, above which radiative losses dominate.
pub const STANDARD_ROCK_CRITICAL_GEV: f32 = STANDARD_ROCK_A / STANDARD_ROCK_B;
