"""Ray construction for parity fixtures.

The direction convention is the forward model's, not an independent choice: a
`Flat2D` bin centred on `phi` produces `(sin phi, cos phi)` (spec §5.3,
`build_system_2d`). Reproducing it here is the point - a ray the Rust side
cannot reconstruct is not a parity test - and the fixture carries `phi` itself
so the Rust test does not have to recover an angle from a direction.
"""

from __future__ import annotations

import math

import numpy as np


def unit_direction(phi: float) -> np.ndarray:
    """`(sin phi, cos phi)`: zenith measured from `+y`, positive toward `+x`."""
    return np.array([math.sin(phi), math.cos(phi)])


def perpendicular(phi: float) -> np.ndarray:
    """In-plane normal to the fan direction, used to offset rays sideways."""
    return np.array([math.cos(phi), -math.sin(phi)])


def fan(centre, angles, offsets, standoff: float) -> list[dict]:
    """Rays aimed at `centre + offset * normal`, starting `standoff` away.

    Each ray carries `phi`, its 2-D origin and direction, and a 3-D padding
    (`z = 0`) so the fixture matches the spec's `{o, d}` triple.
    """
    rays = []
    for phi in angles:
        direction = unit_direction(phi)
        normal = perpendicular(phi)
        for offset in offsets:
            target = np.asarray(centre, dtype=float) + offset * normal
            origin = target - standoff * direction
            rays.append(
                {
                    "phi": float(phi),
                    "o": [float(origin[0]), float(origin[1]), 0.0],
                    "d": [float(direction[0]), float(direction[1]), 0.0],
                }
            )
    return rays


def jittered(
    centre,
    rng,
    count: int,
    standoff: float,
    phi_centre: float = 0.0,
    phi_half_span: float = math.pi,
    offset_half_span: float = 2.0,
) -> list[dict]:
    """Seeded rays at awkward angles, aimed inside the scene half-width.

    These exist so the fixture is not only the tidy angle/offset comb; the seed
    is recorded in the fixture, so regenerating reproduces them exactly.

    The spans narrow the fan for small, finely gridded scenes, where a uniformly
    random `phi` would mostly produce grazing rays that no voxel field can
    represent. The defaults are the full spans the coarse fixtures use, and they
    draw from the generator in the same order, so those fixtures are unchanged.
    """
    rays = []
    for _ in range(count):
        phi = float(rng.uniform(phi_centre - phi_half_span, phi_centre + phi_half_span))
        offset = float(rng.uniform(-offset_half_span, offset_half_span))
        direction = unit_direction(phi)
        target = np.asarray(centre, dtype=float) + offset * perpendicular(phi)
        origin = target - standoff * direction
        rays.append(
            {
                "phi": phi,
                "o": [float(origin[0]), float(origin[1]), 0.0],
                "d": [float(direction[0]), float(direction[1]), 0.0],
            }
        )
    return rays
