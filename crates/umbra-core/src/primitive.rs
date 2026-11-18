//! Analytic primitives.
//!
//! Scenes, hidden voids and detection templates are all described with these
//! shapes so the analytic chord model (A7) can compute ray–primitive
//! intersections in closed form, with no re-voxelisation.

/// A rotation quaternion, stored as `(w, x, y, z)`.
///
/// Kept in-crate so the compute core stays dependency-free and identical on
/// native and `wasm32` targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Constructs a quaternion from its components.
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A solid used for scene geometry, void templates or both.
///
/// All lengths are metres; `c` is a centre, `half` and `semi` are half-extents
/// and semi-axes respectively.
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    /// An oriented box.
    Box {
        c: [f32; 3],
        half: [f32; 3],
        rot: Quat,
    },
    /// An oriented ellipsoid.
    Ellip {
        c: [f32; 3],
        semi: [f32; 3],
        rot: Quat,
    },
    /// A capsule, used for corridors: the swept volume of a sphere of radius
    /// `r` between points `a` and `b`.
    Capsule { a: [f32; 3], b: [f32; 3], r: f32 },
    /// A 2-D extruded polygon, used in slice mode.
    Prism2D {
        /// Polygon vertices in world `(x, y)`, in metres.
        polygon: Vec<[f32; 2]>,
        /// Extrusion range along the out-of-plane axis, in metres.
        extent: [f32; 2],
    },
}

impl Primitive {
    /// Whether `point` lies inside this solid in the slice plane.
    ///
    /// Rotations are **ignored**, matching `umbra-forward::chords::chord_len`:
    /// slice scenes are authored axis-aligned, and a predicate that rotated
    /// while the chord model did not would describe two different objects. The
    /// boundary counts as inside, so a scene's faces land on the voxel centres
    /// that touch them.
    pub fn contains_xy(&self, point: [f32; 2]) -> bool {
        match self {
            Primitive::Box { c, half, .. } => {
                (point[0] >= c[0] - half[0])
                    && (point[0] <= c[0] + half[0])
                    && (point[1] >= c[1] - half[1])
                    && (point[1] <= c[1] + half[1])
            }
            Primitive::Ellip { c, semi, .. } => {
                if semi[0] <= 0.0 || semi[1] <= 0.0 {
                    return false;
                }
                let dx = (point[0] - c[0]) / semi[0];
                let dy = (point[1] - c[1]) / semi[1];
                dx * dx + dy * dy <= 1.0
            }
            Primitive::Capsule { a, b, r } => {
                if *r <= 0.0 {
                    return false;
                }
                // Distance to the segment ab, in the slice plane.
                let (ax, ay) = (point[0] - a[0], point[1] - a[1]);
                let (bx, by) = (b[0] - a[0], b[1] - a[1]);
                let len2 = bx * bx + by * by;
                let t = if len2 <= 0.0 {
                    0.0
                } else {
                    ((ax * bx + ay * by) / len2).clamp(0.0, 1.0)
                };
                let (dx, dy) = (ax - t * bx, ay - t * by);
                dx * dx + dy * dy <= r * r
            }
            Primitive::Prism2D { polygon, .. } => {
                if polygon.len() < 3 {
                    return false;
                }
                let count = polygon.len() as f32;
                let centroid = polygon.iter().fold([0.0f32, 0.0], |acc, vertex| {
                    [acc[0] + vertex[0] / count, acc[1] + vertex[1] / count]
                });
                for index in 0..polygon.len() {
                    let a = polygon[index];
                    let b = polygon[(index + 1) % polygon.len()];
                    let edge = [b[0] - a[0], b[1] - a[1]];
                    let mut normal = [edge[1], -edge[0]];
                    let length = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
                    if length < 1e-12 {
                        continue;
                    }
                    normal = [normal[0] / length, normal[1] / length];
                    if (centroid[0] - a[0]) * normal[0] + (centroid[1] - a[1]) * normal[1] > 0.0 {
                        normal = [-normal[0], -normal[1]];
                    }
                    if (point[0] - a[0]) * normal[0] + (point[1] - a[1]) * normal[1] > 0.0 {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Axis-aligned bounding box in the slice plane, as `(lower, upper)`.
    pub fn bbox_xy(&self) -> ([f32; 2], [f32; 2]) {
        match self {
            Primitive::Box { c, half, .. } => (
                [c[0] - half[0], c[1] - half[1]],
                [c[0] + half[0], c[1] + half[1]],
            ),
            Primitive::Ellip { c, semi, .. } => (
                [c[0] - semi[0], c[1] - semi[1]],
                [c[0] + semi[0], c[1] + semi[1]],
            ),
            Primitive::Capsule { a, b, r } => (
                [a[0].min(b[0]) - r, a[1].min(b[1]) - r],
                [a[0].max(b[0]) + r, a[1].max(b[1]) + r],
            ),
            Primitive::Prism2D { polygon, .. } => {
                let mut lower = [f32::INFINITY, f32::INFINITY];
                let mut upper = [f32::NEG_INFINITY, f32::NEG_INFINITY];
                for vertex in polygon {
                    for axis in 0..2 {
                        lower[axis] = lower[axis].min(vertex[axis]);
                        upper[axis] = upper[axis].max(vertex[axis]);
                    }
                }
                if polygon.is_empty() {
                    ([0.0, 0.0], [0.0, 0.0])
                } else {
                    (lower, upper)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_at(half: f32) -> Primitive {
        Primitive::Box {
            c: [1.0, -2.0, 0.0],
            half: [half, half, 0.0],
            rot: Quat::IDENTITY,
        }
    }

    #[test]
    fn a_box_contains_its_centre_and_its_face() {
        let solid = box_at(1.0);
        assert!(solid.contains_xy([1.0, -2.0]));
        assert!(solid.contains_xy([2.0, -2.0]), "the face is inside");
        assert!(!solid.contains_xy([2.5, -2.0]));
    }

    #[test]
    fn an_ellipse_is_not_a_box() {
        let solid = Primitive::Ellip {
            c: [0.0, 0.0, 0.0],
            semi: [2.0, 1.0, 1.0],
            rot: Quat::IDENTITY,
        };
        assert!(solid.contains_xy([1.9, 0.0]));
        assert!(
            !solid.contains_xy([1.9, 0.9]),
            "outside the ellipse but inside its box"
        );
        assert!(solid.contains_xy([0.0, 1.0]));
    }

    #[test]
    fn a_capsule_is_the_swept_disc() {
        let solid = Primitive::Capsule {
            a: [-3.0, 0.0, 0.0],
            b: [3.0, 0.0, 0.0],
            r: 0.5,
        };
        assert!(solid.contains_xy([0.0, 0.4]));
        assert!(
            solid.contains_xy([3.4, 0.0]),
            "the cap extends past the endpoint"
        );
        assert!(!solid.contains_xy([0.0, 0.6]));
        assert!(!solid.contains_xy([4.0, 0.0]));
    }

    #[test]
    fn a_prism_contains_its_interior_and_not_its_bbox_corners() {
        let solid = Primitive::Prism2D {
            polygon: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            extent: [-1.0, 1.0],
        };
        assert!(solid.contains_xy([0.2, 0.2]));
        assert!(
            !solid.contains_xy([1.8, 1.8]),
            "the hypotenuse corner is outside"
        );
        assert!(!solid.contains_xy([2.5, 0.5]));
    }

    #[test]
    fn bounding_boxes_cover_the_shape() {
        let solid = Primitive::Capsule {
            a: [-3.0, 1.0, 0.0],
            b: [3.0, -1.0, 0.0],
            r: 0.5,
        };
        let (lower, upper) = solid.bbox_xy();
        assert!((lower[0] + 3.5).abs() < 1e-6 && (upper[0] - 3.5).abs() < 1e-6);
        assert!((lower[1] + 1.5).abs() < 1e-6 && (upper[1] - 1.5).abs() < 1e-6);
    }
}
