//! The JavaScript boundary.
//!
//! This crate marshals data and nothing else: every algorithm lives in the
//! crates it re-exports, where it can be tested natively. Task entry points
//! (`BUILD_SYSTEM`, `RECON`, `SCAN`, …), transferable typed-array views and
//! progress events arrive with the ComputeBus ticket; today the façade carries
//! the version handshake the TypeScript client asserts on startup.
//!
//! Bulk data must never cross this boundary through serde: large arrays are
//! handed over as transfers, per spec §4.6.

use wasm_bindgen::prelude::*;

/// Version of the compute-core API contract.
///
/// The TypeScript client compares this against its own expectation at startup,
/// so a stale cached `.wasm` fails loudly instead of silently mismatching.
pub const API_VERSION: &str = "0.1.0";

/// Returns the API version this build speaks.
#[wasm_bindgen]
pub fn api_version() -> String {
    API_VERSION.to_string()
}

/// Whether the client's expected version matches this build.
#[wasm_bindgen]
pub fn version_matches(expected: &str) -> bool {
    expected == API_VERSION
}

/// Entry point for the number of workers the pool should use.
///
/// Kept native so the same policy is unit-testable: `hardwareConcurrency − 1`,
/// clamped to `[1, 4]`, which is the worker-pool size in spec §4.4.
pub fn worker_count(hardware_concurrency: u32) -> u32 {
    hardware_concurrency.saturating_sub(1).clamp(1, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_handshake_round_trips() {
        assert_eq!(api_version(), API_VERSION);
        assert!(version_matches(API_VERSION));
        assert!(!version_matches("0.0.1"));
    }

    #[test]
    fn worker_pool_is_clamped() {
        assert_eq!(worker_count(1), 1);
        assert_eq!(worker_count(8), 4);
        assert_eq!(worker_count(4), 3);
    }
}
