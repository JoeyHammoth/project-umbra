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

Two grid scales are used, and they can assert different things
--------------------------------------------------------------

The coarse scenes (h = 0.5 m, 24 x 24) are what the traversal parity test is
built around: they span a whole scene, and every ray in them must reproduce the
rasterised integral to 1e-4.

The fine scenes (h = 0.01 m) exist to make an analytic chord assertable at all.
At that resolution a curved boundary and a straight one still differ in what can
be claimed:

* ``prism_fine`` - straight edges, crossed near-transversely. A cell-misplaced
  edge moves the chord by at most half a cell, so the per-ray gap is asserted at
  5e-3.
* ``ellipse_fine`` - a curved boundary. The tip of an ellipse shifts by up to a
  whole cell between resolutions, which on a 1.4 m chord is ~1e-2 at h = 1 cm
  however correct the traversal is. That is quantisation, not error, and no
  fixture small enough to commit shrinks it below 5e-3 ray by ray. What *is*
  asserted is the mean signed gap (``analytic_mean_tolerance``), because the
  systematic part - a half-voxel shift, transposed axes, a dropped border cell -
  moves every ray the same way, while quantisation noise cancels over the fan.
  The per-ray worst stays recorded in ``analytic_gap_measured``, so a change in
  it is visible on regeneration even though it is not asserted.

The fine cases aim their rays along one axis on purpose. A chord comparison is
only meaningful where the boundary is crossed transversely: a ray grazing a
boundary spends an unbounded part of its length inside the wrong cell, and the
relative gap there measures the ray choice, not the traversal.

Usage (from `tools/oracle`):

    python -m umbra_oracle fixtures --out ../../crates/umbra-forward/tests/fixtures/oracle --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy

from .rays import fan, jittered
from .shapes import (
    CM_PER_M,
    Box,
    Circle,
    Ellipse,
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

VOXEL_TOLERANCE = 1e-4

# Rays passing through less than this much material (g cm^-2, ~4 um of rock) are
# excluded from the analytic comparison: a grazing chord is smaller than one
# voxel and the rasterised field cannot represent it at all, so the relative gap
# there is 1.0 by construction and says nothing about traversal.
ANALYTIC_FLOOR = 1e-2

FINE_TOLERANCE = 5e-3

# Horizontal-ish rays for the fine grids: straight up is phi = 0, so phi = pi/2
# is the +x direction the fine scenes are elongated along. The tilt is kept small
# on purpose. A ray tilted by delta drifts by delta * chord over its length, and
# if that drift carries it onto a long edge - the parallel ones - the boundary is
# crossed at grazing incidence, where a half-cell misplacement projects onto the
# ray as 0.5h/sin(delta). At delta = 0.12 that is a tenfold amplification of the
# quantisation, so such a ray measures the ray choice and not the traversal.
ELLIPSE_ANGLES = [math.pi / 2 + delta for delta in (-0.04, -0.02, 0.0, 0.02, 0.04)]
PRISM_ANGLES = [math.pi / 2 + delta for delta in (-0.02, -0.01, 0.0, 0.01, 0.02)]


@dataclass(frozen=True)
class Case:
    """One scene, the grid it is rasterised on, and the rays that probe it."""

    name: str
    shapes: list
    analytic_tolerance: float | None
    grid: dict = field(default_factory=lambda: dict(GRID))
    angles: list = field(default_factory=lambda: list(ANGLES))
    offsets: list = field(default_factory=lambda: list(OFFSETS))
    standoff: float = STANDOFF
    length: float = LENGTH
    jitter_count: int = 12
    jitter_offset: float = 2.0
    jitter_phi_centre: float = 0.0
    jitter_phi_half_span: float = math.pi
    # Where the fan is aimed. Defaults to the grid centre, which is where the
    # coarse scenes place their shapes.
    aim: tuple | None = None
    analytic_mean_tolerance: float | None = None


CASES = [
    Case(
        name="box",
        shapes=[Box(-3.0, -1.0, -1.0, 1.0), Box(0.5, 2.5, -0.5, 1.5)],
        analytic_tolerance=1e-5,
    ),
    # Curved and oblique boundaries at h = 0.5 m: the analytic chord is recorded,
    # but the voxelised shape differs from it by percent, so no analytic
    # tolerance is declared and the parity claimed for these cases is the voxel
    # comparison.
    Case(
        name="circles",
        shapes=[Circle(-2.5, 0.0, 1.25), Circle(2.0, 0.5, 1.0)],
        analytic_tolerance=None,
    ),
    Case(
        name="triangle",
        shapes=[Polygon([(0.5, -1.5), (3.5, -1.5), (0.5, 1.5)])],
        analytic_tolerance=None,
    ),
    # Fine grids: 1 cm voxels across a 1-2 m scene. The ray offsets stay well
    # inside the shapes so every chord crosses a boundary transversely, and the
    # grid extends at least ten cells past every vertex, so no part of a shape is
    # clipped by the field the ray is integrated against.
    Case(
        name="ellipse_fine",
        shapes=[Ellipse(0.0, 0.0, 0.7, 0.22)],
        analytic_tolerance=None,
        grid={"n": [160, 50], "origin": [-0.8, -0.25], "h_m": 0.01},
        angles=ELLIPSE_ANGLES,
        offsets=[-0.022, -0.011, 0.0, 0.011, 0.022],
        standoff=1.0,
        length=3.0,
        jitter_count=6,
        jitter_offset=0.022,
        jitter_phi_centre=math.pi / 2,
        jitter_phi_half_span=0.04,
        analytic_mean_tolerance=FINE_TOLERANCE,
    ),
    Case(
        name="prism_fine",
        shapes=[
            Polygon([(-0.85, -0.08), (0.95, -0.08), (0.85, 0.10), (-0.95, 0.10)])
        ],
        analytic_tolerance=FINE_TOLERANCE,
        grid={"n": [260, 36], "origin": [-1.04, -0.144], "h_m": 0.008},
        angles=PRISM_ANGLES,
        offsets=[-0.03, -0.015, 0.0, 0.015, 0.03],
        standoff=1.0,
        length=3.5,
        jitter_count=6,
        jitter_offset=0.03,
        jitter_phi_centre=math.pi / 2,
        jitter_phi_half_span=0.02,
        aim=(0.0, 0.01),
    ),
]


def build_case(case: Case, seed: int) -> dict:
    n = (case.grid["n"][0], case.grid["n"][1])
    origin = np.asarray(case.grid["origin"], dtype=float)
    h = case.grid["h_m"]
    centre = (
        np.asarray(case.aim, dtype=float)
        if case.aim is not None
        else origin + np.asarray(n, dtype=float) * h / 2.0
    )

    rng = np.random.default_rng(seed)
    rays = fan(centre, case.angles, case.offsets, case.standoff) + jittered(
        centre,
        rng,
        case.jitter_count,
        case.standoff,
        phi_centre=case.jitter_phi_centre,
        phi_half_span=case.jitter_phi_half_span,
        offset_half_span=case.jitter_offset,
    )

    field = rasterise(case.shapes, n, origin, h, RHO)
    flat_density = [float(value) for value in field.reshape(-1)]

    analytic = []
    voxel = []
    for ray in rays:
        o = np.asarray(ray["o"][:2], dtype=float)
        d = np.asarray(ray["d"][:2], dtype=float)
        analytic.append(analytic_opacity(case.shapes, RHO, o, d, case.length, QUAD_EPS))
        voxel.append(voxel_opacity(field, n, origin, h, o, d, case.length))

    # The rasterised field and the analytic shapes must agree wherever the shape
    # is representable exactly; where they cannot (curves, slant edges) the case
    # tolerance says how far apart they are allowed to be, and the gap is
    # reported so a change in it is visible when the fixtures are regenerated.
    gaps = []
    for a, v in zip(analytic, voxel):
        if abs(a) > ANALYTIC_FLOOR:
            gaps.append((v - a) / abs(a))
    worst = max((abs(gap) for gap in gaps), default=0.0)

    if case.analytic_tolerance is not None and worst > case.analytic_tolerance:
        raise SystemExit(
            f"{case.name}: analytic and voxel opacities differ by {worst:.2e}, "
            f"more than the {case.analytic_tolerance:.0e} this case declares"
        )

    mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
    if case.analytic_mean_tolerance is not None:
        if not gaps:
            raise SystemExit(
                f"{case.name}: no ray clears the {ANALYTIC_FLOOR:.0e} g/cm2 analytic "
                "floor, so the mean gap it declares cannot be computed"
            )
        if abs(mean_gap) > case.analytic_mean_tolerance:
            raise SystemExit(
                f"{case.name}: mean signed gap {mean_gap:+.2e} over {len(gaps)} rays, "
                f"more than the {case.analytic_mean_tolerance:.0e} this case declares"
            )

    return {
        "schema": SCHEMA,
        "case": case.name,
        "grid": {
            "n": list(n),
            "origin": [float(v) for v in origin],
            "h_m": h,
        },
        "rho_g_cm3": RHO,
        "density": flat_density,
        "rays": rays,
        "expected_opacity": analytic,
        "expected_voxel_opacity": voxel,
        "analytic_tolerance": case.analytic_tolerance,
        "analytic_floor": ANALYTIC_FLOOR,
        "analytic_gap_measured": worst,
        "analytic_mean_tolerance": case.analytic_mean_tolerance,
        "analytic_mean_gap_measured": mean_gap,
        "voxel_tolerance": VOXEL_TOLERANCE,
        "quad_eps": QUAD_EPS,
        "max_path_m": case.length,
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
    for case in CASES:
        fixture = build_case(case, seed)
        path = out / f"{case.name}.json"
        # newline="\n" deliberately: the fixtures are committed with LF, and a
        # Windows regeneration must not rewrite them as CRLF and show a
        # whitespace-only diff that hides real changes.
        path.write_text(
            json.dumps(fixture, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        total_rays += len(fixture["rays"])
        print(
            f"wrote {path} ({len(fixture['rays'])} rays, "
            f"worst gap {fixture['analytic_gap_measured']:.2e} "
            f"asserted at {fixture['analytic_tolerance']}, "
            f"mean gap {fixture['analytic_mean_gap_measured']:+.2e} "
            f"asserted at {fixture['analytic_mean_tolerance']})"
        )
    print(f"total: {total_rays} ray/scene pairs, seed {seed}, schema {SCHEMA}")
    return 0
