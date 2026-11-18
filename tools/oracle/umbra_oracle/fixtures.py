"""Emit parity fixtures for the forward model (spec §6.3).

A fixture describes one scene, its voxel grid, the rasterised density field and a
set of rays, together with two expected opacities per ray:

* ``expected_opacity`` - quadrature of the analytic primitives along the ray. It
  is only as exact as the rasterisation of the shape: a box whose faces sit on
  voxel boundaries reproduces it to f32 noise, whereas a curved or oblique
  boundary at h = 0.5 m is a staircase and misses by percent, not by 1e-4. Each
  case therefore declares ``analytic_tolerance``, or ``null`` where the grid is
  too coarse for that comparison to mean anything - the measured gap is still
  recorded, so the number is not hidden, it is simply not asserted.
* ``expected_voxel_opacity`` - the same ray integrated against the rasterised
  field, which is what the Rust builder must reproduce on *every* ray. This is
  the 1e-4 comparison the parity test is built around.

Neither number comes from Rust. ``self_test`` runs before anything is written and
refuses to emit fixtures if the geometry and the quadrature disagree, so a wrong
predicate cannot be baked into a committed file.

Usage (from `tools/oracle`):

    python -m umbra_oracle fixtures --out ../../crates/umbra-forward/tests/fixtures/oracle --seed 7
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from .rays import fan, jittered
from .shapes import (
    CM_PER_M,
    Box,
    Circle,
    Polygon,
    analytic_opacity,
    rasterise,
    self_test,
    voxel_opacity,
)

SCHEMA = "umbra.oracle/1"
GRID = {"n": [24, 24], "origin": [-6.0, -6.0], "h_m": 0.5}
RHO = 2.65
STANDOFF = 12.0
LENGTH = 24.0
QUAD_EPS = 1e-11

ANGLES = [
    0.0,
    0.3927,  # pi/8
    -0.3927,
    0.7854,  # pi/4
    -0.7854,
    1.0472,  # pi/3
    2.3562,  # 3pi/4
    -2.3562,
]
OFFSETS = [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0]

# (case name, solids, tolerance against the analytic chord)
CASES = [
    (
        "box",
        [Box(-3.0, -1.0, -1.0, 1.0), Box(0.5, 2.5, -0.5, 1.5)],
        1e-5,
    ),
    # Curved and oblique boundaries: the analytic chord is recorded, but at
    # h = 0.5 m the voxelised shape differs from it by percent, so no analytic
    # tolerance is declared and the parity claimed for these cases is the voxel
    # comparison below.
    (
        "circles",
        [Circle(-2.5, 0.0, 1.25), Circle(2.0, 0.5, 1.0)],
        None,
    ),
    (
        "triangle",
        [Polygon([(0.5, -1.5), (3.5, -1.5), (0.5, 1.5)])],
        None,
    ),
]

VOXEL_TOLERANCE = 1e-4

# Rays passing through less than this much material (g cm^-2, ~4 um of rock) are
# excluded from the analytic comparison: a grazing chord is smaller than one
# voxel and the rasterised field cannot represent it at all, so the relative gap
# there is 1.0 by construction and says nothing about traversal.
ANALYTIC_FLOOR = 1e-2


def build_case(name: str, shapes, analytic_tolerance: float, seed: int) -> dict:
    n = (GRID["n"][0], GRID["n"][1])
    origin = np.asarray(GRID["origin"], dtype=float)
    h = GRID["h_m"]
    centre = origin + np.asarray(n) * h / 2.0

    rng = np.random.default_rng(seed)
    rays = fan(centre, ANGLES, OFFSETS, STANDOFF) + jittered(centre, rng, 12, STANDOFF)

    field = rasterise(shapes, n, origin, h, RHO)
    flat_density = [float(value) for value in field.reshape(-1)]

    analytic = []
    voxel = []
    for ray in rays:
        o = np.asarray(ray["o"][:2], dtype=float)
        d = np.asarray(ray["d"][:2], dtype=float)
        analytic.append(analytic_opacity(shapes, RHO, o, d, LENGTH, QUAD_EPS))
        voxel.append(voxel_opacity(field, n, origin, h, o, d, LENGTH))

    # The rasterised field and the analytic shapes must agree wherever the shape
    # is representable exactly; where they cannot (curves, slant edges) the case
    # tolerance says how far apart they are allowed to be, and the gap is
    # reported so a change in it is visible when the fixtures are regenerated.
    worst = 0.0
    for a, v in zip(analytic, voxel):
        if abs(a) > ANALYTIC_FLOOR:
            worst = max(worst, abs(v - a) / abs(a))
    if analytic_tolerance is not None and worst > analytic_tolerance:
        raise SystemExit(
            f"{name}: analytic and voxel opacities differ by {worst:.2e}, "
            f"more than the {analytic_tolerance:.0e} this case declares"
        )

    return {
        "schema": SCHEMA,
        "case": name,
        "grid": {"n": list(n), "origin": [float(v) for v in origin], "h_m": h},
        "rho_g_cm3": RHO,
        "density": flat_density,
        "rays": rays,
        "expected_opacity": analytic,
        "expected_voxel_opacity": voxel,
        "analytic_tolerance": analytic_tolerance,
        "analytic_floor": ANALYTIC_FLOOR,
        "analytic_gap_measured": worst,
        "voxel_tolerance": VOXEL_TOLERANCE,
        "quad_eps": QUAD_EPS,
        "max_path_m": LENGTH,
        "generated_by": {
            "command": f"python -m umbra_oracle fixtures --seed {seed}",
            "seed": seed,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="umbra_oracle")
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("fixtures", help="write parity fixtures")
    emit.add_argument("--out", type=Path, required=True)
    emit.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.command == "fixtures":
        return emit_fixtures(args.out, args.seed)
    raise SystemExit(f"unknown command {args.command}")


def emit_fixtures(out: Path, seed: int) -> int:
    self_test()

    out.mkdir(parents=True, exist_ok=True)
    total_rays = 0
    for name, shapes, analytic_tolerance in CASES:
        fixture = build_case(name, shapes, analytic_tolerance, seed)
        path = out / f"{name}.json"
        path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        total_rays += len(fixture["rays"])
        print(
            f"wrote {path} ({len(fixture['rays'])} rays, "
            f"analytic-vs-voxel gap {fixture['analytic_gap_measured']:.2e}, "
            f"asserted at {fixture['analytic_tolerance']})"
        )
    print(f"total: {total_rays} ray/scene pairs, seed {seed}, schema {SCHEMA}")
    return 0
