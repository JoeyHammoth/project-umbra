"""Ray-primitive geometry and line integrals.

The lab's *semantics* are kept: a solid contributes attenuation inside the shape
and zero outside, and a scene sums its solids, so overlapping material adds. Its
*code* is not: ``source/ctlab`` is never imported and never modified (spec
§13.1), it is the citation for the picture.

The vendored ``Circle.attenuation`` is wrong in two ways and cannot be used as a
reference even by accident: it tests ``(x - x0)**2 + (y**2 - y0)**2 < r**2``,
which is a quartic curve and not a circle, and it reads an undefined ``r`` (the
constructor stores ``self.r``). The intended predicate is
``(x - x0)**2 + (y - y0)**2 <= r**2``, and that is what is implemented here.
``self_test`` below pins it: an off-axis circle whose chord is known in closed
form is compared against both the quadrature and the predicate.

Two independent numbers are produced per ray:

* **analytic** - ``scipy.integrate.quad`` of the indicator over the material
  interval, mirroring ``AttenuationObject.project_attenuation``. The interval
  endpoints come from the closed-form intersection, so the same ray is described
  twice in two ways; the generator refuses to write a fixture when the two
  disagree.
* **voxel** - the same ray against the *rasterised* field, by clipping it
  against every cell box. No traversal, no DDA, no ordering: this is the number
  the Rust builder has to reproduce.

Lengths are metres, densities g cm^-3, and the returned opacities g cm^-2 - the
same units ``System::opacity`` uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

CM_PER_M = 100.0


class Shape:
    """A solid with a point predicate and a closed-form ray interval."""

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def interval(
        self, origin: np.ndarray, direction: np.ndarray, length: float
    ) -> tuple[float, float]:
        """``(t_enter, t_exit)`` inside the shape, clipped to ``[0, length]``."""
        raise NotImplementedError

    def chord(self, origin: np.ndarray, direction: np.ndarray, length: float) -> float:
        t0, t1 = self.interval(origin, direction, length)
        return max(0.0, t1 - t0)


@dataclass
class Box(Shape):
    """Axis-aligned rectangle, `attenuation` 1 inside and 0 outside."""

    x0: float
    x1: float
    y0: float
    y1: float

    def inside(self, x, y):
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)

    def interval(self, origin, direction, length):
        t_enter, t_exit = 0.0, length
        for o, d, lo, hi in (
            (origin[0], direction[0], self.x0, self.x1),
            (origin[1], direction[1], self.y0, self.y1),
        ):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return (0.0, 0.0)
                continue
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
        return (t_enter, t_exit) if t_exit > t_enter else (0.0, 0.0)


@dataclass
class Circle(Shape):
    """Circle with attenuation inside and 0 outside - the corrected predicate."""

    x0: float
    y0: float
    r: float

    def inside(self, x, y):
        return ((x - self.x0) ** 2 + (y - self.y0) ** 2) <= self.r**2

    def interval(self, origin, direction, length):
        px, py = origin[0] - self.x0, origin[1] - self.y0
        a = direction[0] ** 2 + direction[1] ** 2
        b = 2.0 * (px * direction[0] + py * direction[1])
        c = px * px + py * py - self.r**2
        disc = b * b - 4.0 * a * c
        if disc <= 0.0 or a <= 0.0:
            return (0.0, 0.0)
        root = np.sqrt(disc)
        t1 = (-b - root) / (2.0 * a)
        t2 = (-b + root) / (2.0 * a)
        t_enter = max(t1, 0.0)
        t_exit = min(t2, length)
        return (t_enter, t_exit) if t_exit > t_enter else (0.0, 0.0)


@dataclass
class Ellipse(Shape):
    """Axis-aligned ellipse: the slice-plane section of `Primitive::Ellip`.

    There is deliberately no rotation field. Slice mode ignores primitive
    rotations (`chords::chord_len` says so in as many words), so an ellipse that
    could rotate would describe a scene the forward model cannot represent, and
    the fixture would be comparing two different objects.
    """

    x0: float
    y0: float
    semi_x: float
    semi_y: float

    def inside(self, x, y):
        return ((x - self.x0) ** 2 / self.semi_x**2) + (
            (y - self.y0) ** 2 / self.semi_y**2
        ) <= 1.0

    def interval(self, origin, direction, length):
        # Same scaled-quadratic solve the Rust side does, kept independent of it
        # by being written from the algebra rather than transliterated.
        px, py = (origin[0] - self.x0) / self.semi_x, (origin[1] - self.y0) / self.semi_y
        vx, vy = direction[0] / self.semi_x, direction[1] / self.semi_y
        a = vx * vx + vy * vy
        b = 2.0 * (px * vx + py * vy)
        c = px * px + py * py - 1.0
        disc = b * b - 4.0 * a * c
        if disc <= 0.0 or a <= 0.0:
            return (0.0, 0.0)
        root = np.sqrt(disc)
        t1 = (-b - root) / (2.0 * a)
        t2 = (-b + root) / (2.0 * a)
        t_enter = max(t1, 0.0)
        t_exit = min(t2, length)
        return (t_enter, t_exit) if t_exit > t_enter else (0.0, 0.0)


@dataclass
class Polygon(Shape):
    """Convex polygon by half-plane clipping."""

    vertices: list[tuple[float, float]]

    def inside(self, x, y):
        points = np.asarray(self.vertices, dtype=float)
        centroid = points.mean(axis=0)
        result = np.ones(np.broadcast(x, y).shape, dtype=bool)
        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            if np.dot(centroid - a, normal) > 0.0:
                normal = -normal
            result &= (x - a[0]) * normal[0] + (y - a[1]) * normal[1] <= 0.0
        return result

    def interval(self, origin, direction, length):
        points = np.asarray(self.vertices, dtype=float)
        centroid = points.mean(axis=0)
        t_enter, t_exit = 0.0, length
        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            if np.dot(centroid - a, normal) > 0.0:
                normal = -normal
            distance = (origin[0] - a[0]) * normal[0] + (origin[1] - a[1]) * normal[1]
            slope = direction[0] * normal[0] + direction[1] * normal[1]
            if abs(slope) < 1e-12:
                if distance > 0.0:
                    return (0.0, 0.0)
                continue
            t = -distance / slope
            if slope > 0.0:
                t_exit = min(t_exit, t)
            else:
                t_enter = max(t_enter, t)
        return (t_enter, t_exit) if t_exit > t_enter else (0.0, 0.0)


def analytic_opacity(
    shapes: list[Shape],
    rho: float,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    quad_eps: float = 1e-11,
) -> float:
    """Line integral of the indicator, times density, in g cm^-2.

    This is the lab's computation: quadrature of the attenuation along the ray.
    The interval endpoints are passed to ``quad`` so a curved boundary is not
    integrated across its kink at default subdivision.
    """
    total = 0.0
    for shape in shapes:
        t0, t1 = shape.interval(origin, direction, length)
        if t1 <= t0:
            continue

        def indicator(t, shape=shape):
            x = origin[0] + t * direction[0]
            y = origin[1] + t * direction[1]
            return 1.0 if bool(shape.inside(np.array([x]), np.array([y]))[0]) else 0.0

        value, _ = quad(indicator, t0, t1, epsabs=quad_eps, epsrel=quad_eps, limit=200)
        total += value
    return total * CM_PER_M * rho


def rasterise(shapes: list[Shape], n: tuple[int, int], origin, h: float, rho: float):
    """Indicator field on voxel centres, times density.

    Shape ``(n[1], n[0])`` with ``field[j, i]`` at cell ``(i, j)``, which flattens
    in C order to the row-major ``j * n[0] + i`` indexing the Rust grid uses.
    Overlapping solids add, as they do in the lab's ``ObjectCollection``.
    """
    xs = origin[0] + (np.arange(n[0]) + 0.5) * h
    ys = origin[1] + (np.arange(n[1]) + 0.5) * h
    grid_x, grid_y = np.meshgrid(xs, ys)
    field = np.zeros_like(grid_x)
    for shape in shapes:
        field += shape.inside(grid_x, grid_y).astype(float)
    return field * rho


def _axis_slab(o: float, d: float, lo: np.ndarray, hi: np.ndarray, length: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    if abs(d) < 1e-12:
        # Half-open [lo, hi): a ray lying exactly on a cell face belongs to one
        # cell, not both. With `<= hi` a boundary-aligned ray is counted in both
        # neighbouring columns and the integral doubles - which is precisely
        # what this check caught, at exactly 2x on a vertical ray through the
        # grid interior.
        inside = (o >= lo) & (o < hi)
        t_min = np.where(inside, 0.0, np.inf)
        t_max = np.where(inside, length, -np.inf)
    return t_min, t_max


def voxel_opacity(
    field: np.ndarray,
    n: tuple[int, int],
    origin,
    h: float,
    ray_origin: np.ndarray,
    direction: np.ndarray,
    length: float,
) -> float:
    """Exact integral of a piecewise-constant voxel field along a ray, g cm^-2.

    Clips the ray against every cell box and sums ``field * segment length``.
    Deliberately independent of any traversal: if the DDA steps a half voxel
    short, drops an entry cell or mis-orders indices, this disagrees.
    """
    low_x = origin[0] + np.arange(n[0]) * h
    low_y = origin[1] + np.arange(n[1]) * h
    t_xmin, t_xmax = _axis_slab(ray_origin[0], direction[0], low_x, low_x + h, length)
    t_ymin, t_ymax = _axis_slab(ray_origin[1], direction[1], low_y, low_y + h, length)

    t_enter = np.maximum(t_ymin[:, None], t_xmin[None, :])
    t_exit = np.minimum(t_ymax[:, None], t_xmax[None, :])
    t_enter = np.clip(t_enter, 0.0, length)
    t_exit = np.clip(t_exit, 0.0, length)
    segment = np.clip(t_exit - t_enter, 0.0, None)
    return float((segment * field).sum()) * CM_PER_M


def self_test() -> None:
    """Checks that would have caught the vendored circle bug.

    A circle centred off-axis in `y` has a chord of known length; the quadrature,
    the closed form and the point predicate must all agree. `(y**2 - y0)**2`
    instead of `(y - y0)**2` fails this at the first assertion.
    """
    origin = np.array([-5.0, 0.5])
    direction = np.array([1.0, 0.0])
    circle = Circle(0.0, 0.5, 1.0)

    assert bool(circle.inside(np.array([0.0]), np.array([0.5]))[0]), "centre inside"
    assert not bool(circle.inside(np.array([0.0]), np.array([2.0]))[0]), "outside point"

    closed_form = circle.chord(origin, direction, 10.0)
    assert abs(closed_form - 2.0) < 1e-12, f"chord {closed_form}"

    quad_value = analytic_opacity([circle], 1.0, origin, direction, 10.0) / CM_PER_M
    assert abs(quad_value - 2.0) < 1e-9, f"quadrature {quad_value} vs chord {closed_form}"

    box = Box(-1.0, 1.0, -1.0, 1.0)
    diagonal = box.chord(np.array([-5.0, -5.0]), np.array([1.0, 1.0]) / np.sqrt(2.0), 20.0)
    assert abs(diagonal - 2.0 * np.sqrt(2.0)) < 1e-12, f"box diagonal {diagonal}"

    triangle = Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
    across = triangle.chord(np.array([-1.0, 0.5]), direction, 10.0)
    assert abs(across - 1.5) < 1e-12, f"triangle chord {across}"

    ellipse = Ellipse(0.0, 0.0, 2.0, 1.0)
    assert abs(ellipse.chord(np.array([-5.0, 0.0]), direction, 10.0) - 4.0) < 1e-12
    assert (
        abs(ellipse.chord(np.array([0.0, -5.0]), np.array([0.0, 1.0]), 10.0) - 2.0)
        < 1e-12
    )
    # Off-centre and oblique: the closed form and the quadrature of the point
    # predicate must agree. A semi-axis in the wrong place, a missing division by
    # semi**2 or a rotated axis all break this while leaving the centre rays
    # above intact.
    angle = 0.2915
    oblique = np.array([math.cos(angle), math.sin(angle)])
    start = np.array([-5.0, -0.5])
    closed = ellipse.chord(start, oblique, 10.0)
    quadrature = analytic_opacity([ellipse], 1.0, start, oblique, 10.0) / CM_PER_M
    assert closed > 0.5, f"oblique ellipse chord {closed} misses the shape"
    assert abs(closed - quadrature) < 1e-8, f"ellipse {closed} vs quadrature {quadrature}"
