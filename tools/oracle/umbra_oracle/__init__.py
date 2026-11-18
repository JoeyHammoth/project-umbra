"""UMBRA's independent reference oracle (spec §6.3).

Shares no code with the Rust crates, by design: for a line integral, a second
implementation that could reproduce the same mistake is worth nothing. It
supplies the analytic chords and voxel-field integrals that
`crates/umbra-forward/tests/oracle_parity.rs` compares traversal against.

    python -m umbra_oracle fixtures --out crates/umbra-forward/tests/fixtures/oracle --seed 7
"""

__all__ = ["fixtures", "rays", "shapes"]
