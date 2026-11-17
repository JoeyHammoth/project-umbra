use crate::object::AttenuationObject;

/// A circle centered on `(x0, y0)` of radius `r` with constant
/// attenuation factor inside.
pub struct Circle {
    x0: f64,
    y0: f64,
    r: f64,
    attenuation_factor: f64,
}

impl Circle {
    /// Create a new Circle object centered on `(x0, y0)` with radius `r` and
    /// the specified `attenuation_factor`.
    ///
    /// # Example
    ///
    /// ```
    /// use ctlab::object::Circle;
    /// use ctlab::object::AttenuationObject;
    ///
    /// let circle = Circle::new(0., 0., 1., 3.);
    /// let diff = (circle.project_attenuation(0., 0., (-5., 5.)) - 6.).abs();
    /// assert!(diff < 1e-2);
    /// ```
    pub fn new(x0: f64, y0: f64, r: f64, attenuation_factor: f64) -> Circle {
        Circle {
            x0,
            y0,
            attenuation_factor,
            r: r.abs(),
        }
    }
}

// Implement the AttenuationObject trait for Circle.
impl AttenuationObject for Circle {
    fn attenuation(&self, x: f64, y: f64) -> f64 {
        unimplemented!();
    }
}

// Allow Circle to be synced across threads
unsafe impl Sync for Circle {}
