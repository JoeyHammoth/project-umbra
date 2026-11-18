//! Analytic chords through the scene primitives, in the slice plane.
//!
//! This is the independent-geometry half of the parity story (spec §6.3): the
//! voxel builder subdivides, this does not, and the tests compare the two. It is
//! also the forward model A7 uses for void parameters, where re-voxelising on
//! every likelihood evaluation would be far too slow.
//!
//! # Slice-plane convention
//!
//! 2-D mode treats the `xy` plane as the slice: `x` horizontal, `y` vertical, ray
//! directions unit vectors in that plane. Rotations are **ignored** here: slice
//! scenes are authored axis-aligned, and silently rotating a primitive in a mode
//! that cannot represent the third dimension would be worse than saying so.

use umbra_core::Primitive;

/// Length of the chord through `primitive` along the ray `origin + t * dir`.
///
/// `dir` need not be normalised. Returns `0.0` when the ray misses.
pub fn chord_len(primitive: &Primitive, origin: [f32; 2], dir: [f32; 2]) -> f32 {
    let norm = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return 0.0;
    }
    let d = [dir[0] / norm, dir[1] / norm];
    match primitive {
        Primitive::Box { c, half, .. } => box_chord([c[0], c[1]], [half[0], half[1]], origin, d),
        Primitive::Ellip { c, semi, .. } => {
            ellipsoid_chord([c[0], c[1]], [semi[0], semi[1]], origin, d)
        }
        Primitive::Prism2D { polygon, .. } => polygon_chord(polygon, origin, d),
        // A capsule in the slice plane is a capsule; the closed form is a
        // quadratic in a rotated frame. Not needed by M1's scenarios, so this
        // reports zero rather than returning an approximation that would be
        // mistaken for a chord.
        Primitive::Capsule { .. } => 0.0,
    }
}

/// Axis-aligned box chord by slab clipping.
fn box_chord(c: [f32; 2], half: [f32; 2], origin: [f32; 2], dir: [f32; 2]) -> f32 {
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for axis in 0..2 {
        let lo = c[axis] - half[axis];
        let hi = c[axis] + half[axis];
        if dir[axis].abs() < 1e-12 {
            if origin[axis] < lo || origin[axis] > hi {
                return 0.0;
            }
            continue;
        }
        let inv = 1.0 / dir[axis];
        let t1 = (lo - origin[axis]) * inv;
        let t2 = (hi - origin[axis]) * inv;
        t_enter = t_enter.max(t1.min(t2));
        t_exit = t_exit.min(t1.max(t2));
    }
    if t_exit <= t_enter {
        0.0
    } else {
        (t_exit - t_enter).max(0.0)
    }
}

/// Ellipsoid chord by solving the scaled quadratic.
fn ellipsoid_chord(c: [f32; 2], semi: [f32; 2], origin: [f32; 2], dir: [f32; 2]) -> f32 {
    if semi[0] <= 0.0 || semi[1] <= 0.0 {
        return 0.0;
    }
    let p = [(origin[0] - c[0]) / semi[0], (origin[1] - c[1]) / semi[1]];
    let v = [dir[0] / semi[0], dir[1] / semi[1]];
    let a = v[0] * v[0] + v[1] * v[1];
    let b = 2.0 * (p[0] * v[0] + p[1] * v[1]);
    let cc = p[0] * p[0] + p[1] * p[1] - 1.0;
    let disc = b * b - 4.0 * a * cc;
    if disc <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    let root = disc.sqrt();
    let t1 = (-b - root) / (2.0 * a);
    let t2 = (-b + root) / (2.0 * a);
    (t2 - t1).max(0.0)
}

/// Chord through a convex polygon by half-plane clipping.
fn polygon_chord(polygon: &[[f32; 2]], origin: [f32; 2], dir: [f32; 2]) -> f32 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let centroid = polygon.iter().fold([0.0f32, 0.0], |acc, p| {
        [
            acc[0] + p[0] / polygon.len() as f32,
            acc[1] + p[1] / polygon.len() as f32,
        ]
    });

    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let mut normal = [edge[1], -edge[0]];
        let length = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        if length < 1e-12 {
            continue;
        }
        normal = [normal[0] / length, normal[1] / length];
        // Point the normal away from the interior.
        let outward = (centroid[0] - a[0]) * normal[0] + (centroid[1] - a[1]) * normal[1];
        if outward > 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        let distance = (origin[0] - a[0]) * normal[0] + (origin[1] - a[1]) * normal[1];
        let slope = dir[0] * normal[0] + dir[1] * normal[1];
        if slope.abs() < 1e-12 {
            if distance > 0.0 {
                return 0.0;
            }
            continue;
        }
        let t = -distance / slope;
        if slope > 0.0 {
            t_exit = t_exit.min(t);
        } else {
            t_enter = t_enter.max(t);
        }
    }
    if t_exit <= t_enter {
        0.0
    } else {
        (t_exit - t_enter).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::Quat;

    fn unit_box() -> Primitive {
        Primitive::Box {
            c: [0.0, 0.0, 0.0],
            half: [1.0, 1.0, 1.0],
            rot: Quat::IDENTITY,
        }
    }

    #[test]
    fn box_chord_is_the_side_length_straight_through() {
        let len = chord_len(&unit_box(), [0.0, 0.0], [1.0, 0.0]);
        assert!((len - 2.0).abs() < 1e-6, "chord = {len}");
    }

    #[test]
    fn box_chord_follows_the_diagonal() {
        // A unit box, entered on a diagonal, has chord 2*sqrt(2) only through a
        // cube; in 2-D the diagonal chord of a 2x2 square is 2*sqrt(2).
        let len = chord_len(&unit_box(), [-2.0, -2.0], [1.0, 1.0]);
        assert!((len - 2.0 * 2f32.sqrt()).abs() < 1e-5, "chord = {len}");
    }

    #[test]
    fn a_ray_that_misses_returns_zero() {
        assert_eq!(chord_len(&unit_box(), [0.0, 5.0], [1.0, 0.0]), 0.0);
    }

    #[test]
    fn ellipsoid_chord_through_the_middle_is_twice_the_semi_axis() {
        let ellipse = Primitive::Ellip {
            c: [0.0, 0.0, 0.0],
            semi: [2.0, 1.0, 1.0],
            rot: Quat::IDENTITY,
        };
        assert!((chord_len(&ellipse, [-5.0, 0.0], [1.0, 0.0]) - 4.0).abs() < 1e-5);
        assert!((chord_len(&ellipse, [0.0, -5.0], [0.0, 1.0]) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn polygon_chord_matches_a_triangle() {
        // Right triangle with legs 2, entering parallel to the x axis at y = 0.5
        // crosses from x = 0 to x = 1.5, a chord of 1.5.
        let triangle = Primitive::Prism2D {
            polygon: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            extent: [-1.0, 1.0],
        };
        let len = chord_len(&triangle, [-1.0, 0.5], [1.0, 0.0]);
        assert!((len - 1.5).abs() < 1e-5, "chord = {len}");
    }

    #[test]
    fn polygon_chord_is_orientation_agnostic() {
        let ccw = Primitive::Prism2D {
            polygon: vec![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
            extent: [-1.0, 1.0],
        };
        let cw = Primitive::Prism2D {
            polygon: vec![[0.0, 2.0], [4.0, 2.0], [4.0, 0.0], [0.0, 0.0]],
            extent: [-1.0, 1.0],
        };
        let a = chord_len(&ccw, [-1.0, 1.0], [1.0, 0.0]);
        let b = chord_len(&cw, [-1.0, 1.0], [1.0, 0.0]);
        assert!(
            (a - 4.0).abs() < 1e-5 && (a - b).abs() < 1e-5,
            "a={a} b={b}"
        );
    }

    #[test]
    fn capsule_is_explicitly_unsupported_rather_than_approximated() {
        let capsule = Primitive::Capsule {
            a: [0.0, 0.0, 0.0],
            b: [1.0, 0.0, 0.0],
            r: 0.5,
        };
        assert_eq!(chord_len(&capsule, [-5.0, 0.0], [1.0, 0.0]), 0.0);
    }
}
