#!/usr/bin/env python3
"""Independent check of the committed flux table.

Reimplements the Guan et al. (2015) parametrisation (arXiv:1509.06176, Eq. 3
with the curvature angle of Eq. 2) from scratch and compares it against the
serialised asset that the Rust core actually ships. Nothing is imported from the
Rust side, so a sign error, a swapped coefficient or a wrong integral bound
cannot cancel out between the two.

Usage:
    python tools/oracle/check_flux.py [path/to/flux.bin]

Exit code 0 on success, 1 on disagreement. The tolerance is 1%: the table stores
f32 values and interpolates, so agreement at the 0.1% level is expected.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

# Guan et al. (2015), Table 1.
P = (0.102573, -0.068287, 0.958633, 0.0407253, 0.817285)

MAGIC = b"UBT1"
TOLERANCE = 1e-2


def cos_theta_star(cos_theta: float) -> float:
    c = min(max(cos_theta, 0.0), 1.0)
    numerator = np.sqrt(c * c + P[0] ** 2 + P[1] * c ** P[2] + P[3] * c ** P[4])
    denominator = np.sqrt(1.0 + P[0] ** 2 + P[1] + P[3])
    return numerator / denominator


def differential_flux(energy_gev: float, cos_theta: float) -> float:
    """I(E, θ) in cm^-2 s^-1 sr^-1 GeV^-1 (Eq. 3)."""
    if energy_gev <= 0.0:
        return 0.0
    cs = cos_theta_star(cos_theta)
    modified = energy_gev * (1.0 + 3.64 / (energy_gev * cs ** 1.29))
    pion = 1.0 / (1.0 + 1.1 * energy_gev * cs / 115.0)
    kaon = 0.054 / (1.0 + 1.1 * energy_gev * cs / 850.0)
    return 0.14 * modified ** -2.7 * (pion + kaon)


def integral_flux(energy_gev: float, cos_theta: float) -> float:
    """Φ(>E, θ) by adaptive quadrature, in cm^-2 s^-1 sr^-1."""
    value, _ = quad(
        differential_flux,
        energy_gev,
        1.0e6,
        args=(cos_theta,),
        limit=400,
        epsabs=1e-12,
        epsrel=1e-9,
    )
    return value


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    raw = path.read_bytes()
    if raw[:4] != MAGIC:
        raise SystemExit(f"{path}: not a UMBRA table (bad magic {raw[:4]!r})")
    version, _flags = struct.unpack("<HH", raw[4:8])
    if version != 1:
        raise SystemExit(f"{path}: unsupported version {version}")
    n_e, n_cos = struct.unpack("<II", raw[8:16])
    e0, de, c0, dc = struct.unpack("<ffff", raw[16:32])
    count = n_e * n_cos
    integral = np.frombuffer(raw[32 : 32 + 4 * count], dtype="<f4").reshape(n_e, n_cos)
    return integral, integral, e0, de, c0, dc


def bilinear(table: np.ndarray, e0: float, de: float, c0: float, dc: float,
             n_cos: int, energy: float, cos_theta: float) -> float:
    n_e = table.shape[0]
    log_e = np.log10(max(energy, 1e-6))
    pos = (log_e - e0) / de
    i = int(min(max(np.floor(pos), 0), n_e - 2))
    t = min(max(pos - i, 0.0), 1.0)
    cos_theta = min(max(cos_theta, 0.0), 1.0)
    posc = (cos_theta - c0) / dc if dc else 0.0
    j = int(min(max(np.floor(posc), 0), n_cos - 2))
    tc = min(max(posc - j, 0.0), 1.0)
    low = table[i, j] * (1 - t) + table[i + 1, j] * t
    high = table[i, j + 1] * (1 - t) + table[i + 1, j + 1] * t
    return float(low * (1 - tc) + high * tc)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("web/public/tables/v1/flux.bin")
    if not path.exists():
        print(f"FAIL: {path} does not exist - run `cargo run -p umbra-cli -- gen-flux`")
        return 1

    integral, _, e0, de, c0, dc = load_table(path)
    n_cos = integral.shape[1]
    print(f"table: {path} {integral.shape[0]}x{n_cos} nodes, e0={e0} de={de:.6f}")

    failures = 0
    # Energies spanning the grid, at vertical and inclined angles.
    for energy in (0.5, 1.0, 3.0, 10.0, 100.0, 1000.0):
        for cos_theta in (1.0, 0.7, 0.3):
            expected = integral_flux(energy, cos_theta)
            stored = bilinear(integral, e0, de, c0, dc, n_cos, energy, cos_theta)
            relative = abs(stored - expected) / expected
            status = "ok" if relative < TOLERANCE else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"  {status:4s} E={energy:7.1f} GeV cos={cos_theta:.1f} "
                f"table={stored:.4e} quadrature={expected:.4e} rel={relative:.2e}"
            )

    # The literature anchor that caught the original sign error.
    anchor = integral_flux(1.0, 1.0) * 1.0e4
    print(f"\nanchor: Phi(>1 GeV, vertical) = {anchor:.1f} m^-2 s^-1 sr^-1 (literature ~70)")
    if not 50.0 <= anchor <= 90.0:
        print("FAIL: vertical flux above 1 GeV is outside 50-90 m^-2 s^-1 sr^-1")
        failures += 1

    if failures:
        print(f"\nFAILED ({failures} disagreement(s))")
        return 1
    print("\nPASS: table agrees with the independent quadrature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
