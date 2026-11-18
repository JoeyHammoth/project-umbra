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
