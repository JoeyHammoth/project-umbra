use crate::object::AttenuationObject;

/// A rectangle in `(x, y)` coordinates with constant attenuation factor inside.
pub struct Rectangle {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    attenuation_factor: f64,
}

impl Rectangle {
    /// Create a new Rectangle object bounded by `x1` and `x2` in the `x`-axis,
    /// and `y1` and `y2` in the `y`-axis.
    ///
    /// # Example
    ///
    /// ```
    /// use ctlab::object::Rectangle;
    /// use ctlab::object::AttenuationObject;
    ///
    /// let rect = Rectangle::new((-3., 3.), (-2., 1.), 3.);
    /// let diff = (rect.project_attenuation(0., 0., (-5., 5.)) - 9.).abs();
    /// assert!(diff < 1e-2);
    /// ```
    pub fn new((x1, x2): (f64, f64), (y1, y2): (f64, f64), attenuation_factor: f64) -> Rectangle {
        let (x1, x2) = if x1 > x2 { (x2, x1) } else { (x1, x2) };
        let (y1, y2) = if y1 > y2 { (y2, y1) } else { (y1, y2) };

        Rectangle {
            x1,
            x2,
            y1,
            y2,
            attenuation_factor,
        }
    }
}

// Implement the AttenuationObject trait for Rectangle.
impl AttenuationObject for Rectangle {
    fn attenuation(&self, x: f64, y: f64) -> f64 {
        unimplemented!();
    }
}

// Allow Rectangle to be synced across threads
unsafe impl Sync for Rectangle {}
