//! When taking projection of objects, the object that is being projected is
//! arbitrary.  It could be a nice geometrical object, some function ℝ² → ℝ, an
//! image, etc.  In terms of programming, the handling of each instance on a
//! base-by-case basis is not only cumbersome, but will lead to a lot of code
//! reduplication which we would rather avoid.
//!
//! Fortunately, most modern programming languages provide a way to handle this.
//! In Rust, this is through the use of *traits*.  A trait is just a collection
//! of functions (and possibly constants) which are guaranteed to come together.
//! In this instance, the trait is `AttenuationObject` and consists of the
//! functions `attenuation`, `project_attenuation` and `to_array`.
//!
//! With the trait defined, we can now define functions that take *any* argument
//! provided it implements the specific trait(s) specified.  Take for example
//! this code snippet:
//!
//! ```
//! use ctlab::object::AttenuationObject;
//!
//! fn attenuation_at_origin<T>(object: &T) -> f64
//!     where T: AttenuationObject
//! {
//!     let att = object.attenuation(0., 0.);
//!     println!("Attenuation at the origin is {}.", att);
//!     att
//! }
//!
//! assert_eq!(attenuation_at_origin(&|x, y| x * y), 0.);
//! ```
//!
//! The `<T>` after the function name specifies that we have some type that
//! isn't specified and is left arbitrary.  The line `where T: AttenutionObject`
//! then specifies that `T` can be any type provided that it implements the
//! trait `AttenuationObject`.  Since the compiler knows that whatever the
//! `object` is will have the `AttenuationObject` trait, then we can call
//! `object.attenuation()`.  In the example above, it simply gets the
//! attenuation at the origin and prints it.

mod circle;
mod object_collection;
mod rectangle;

pub use self::circle::Circle;
pub use self::object_collection::ObjectCollection;
pub use self::rectangle::Rectangle;

use crate::integration;
use image::{DynamicImage, GenericImageView, Pixel};
use ndarray::{Array1, Array2};

/// The generic traits of an attenuation object.
///
/// This defines which methods an attenuation object must have, and possibly
/// defines a default implementation of these methods if it is possible.
pub trait AttenuationObject {
    /// Return the attenuation factor μ at the point `(x, y)` in the
    /// original Cartesian coordinates.
    ///
    /// Note that this function is declared but *not* defined (in the source
    /// code, it does not have the braces that say what the function does).
    /// This is because the `AttenuationObject` trait requires this function to
    /// exist, but it doesn't care how it is implemented.
    fn attenuation(&self, x: f64, y: f64) -> f64;

    /// Return the integrated attenuation factor for a given xi and theta value.
    /// The integration is done along the specified range of eta values.
    fn project_attenuation(&self, theta: f64, xi: f64, eta_range: (f64, f64)) -> f64 {
        // Although we don't know yet what the specifies of the attenuation
        // object, we know that we can get the attenuation at any `(x, y)`
        // coordinate.  This is sufficient for us to implement
        // `project_attenuation`.
        //
        // Note that this definition of `project_attenuation` can (but does not
        // *need* to) be defined by the specific implementation.  For example,
        // one might have have on object that has a constant attenuation
        // everywhere in which case projecting the attenuation is trivial.

        unimplemented!();
    }

    /// Create a 2D array in the `(x, y)` coordinate with the entries
    /// corresponding the the attenuation at the location.  No projection is
    /// done here, this function just creates a 2D 'image' of the projection
    /// over the specified range.
    fn to_array(&self, x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
        Array2::from_shape_fn((x.len(), y.len()), |(xi, yi)| {
            self.attenuation(x[xi], y[yi])
        })
    }
}

/// Implement the `AttenuationObject` for any function ℝ² → ℝ.
///
/// This can be useful if you want your attenuation has a nice analytic form.
/// For example, a 2D standard Gaussian attenuation factor would be:
///
/// ```
/// use ctlab::object::AttenuationObject;
/// use std::f64::consts::PI;
///
/// fn attenuation_at_origin<T>(obj: &T) -> f64
/// where
///     T: AttenuationObject,
/// {
///     let att = obj.attenuation(0.0, 0.0);
///     println!("Attenuation at origin is: {}", att);
///     att
/// }
///
/// let gaussian = |x: f64, y: f64| ((- x.powi(2) - y.powi(2)) / 2.0).exp() / (2.0 * PI);
///
/// assert_eq!(attenuation_at_origin(&gaussian), 1.0 / (2.0 * PI));
/// ```
///
/// Since defining a new function can be cumbersome, we use a *closure*
/// (a.k.a. a lambda function).  This provides a quick way of defining a
/// function where the arguments start at `|...|` and the body of the function
/// are in the braces following it (the braces are optional if the function is a
/// one-liner, as above).  In general, it isn't necessarily to specify the type
/// of the inputs (`f64` in the above case).
impl<T> AttenuationObject for T
where
    T: Fn(f64, f64) -> f64,
{
    fn attenuation(&self, x: f64, y: f64) -> f64 {
        self(x, y)
    }
}

/// Implement the `AttenuationObject` for images too.  The image is rescaled so
/// that if fits within the `[-1, 1]^2`.  For a square image, this means that
/// the coordinate `(-1, -1)` will correspond to the lower left corner, and the
/// coordinate `(1, 1)` corresponds to the upper right corner.
///
/// This is all implemented for you and you need not do anything.
impl AttenuationObject for DynamicImage {
    fn attenuation(&self, x: f64, y: f64) -> f64 {
        // Get the bounding rectangle for the image, and convert to f64.
        let (ix, iy, iw, ih) = self.bounds();
        let (ix, iy, iw, ih) = (f64::from(ix), f64::from(iy), f64::from(iw), f64::from(ih));

        // We scale the input (x, y) coordinates so the correspond to a pixel
        // index.
        let scale_factor = if iw > ih { iw / 2.0 } else { ih / 2.0 };
        let (x, y) = (scale_factor * x, scale_factor * y);
        let (x, y) = (x + ix + iw / 2.0, y + iy + ih / 2.0);

        // Convert the new coordinate into a pixel coordinate
        let (x, y) = (x as u32, y as u32);

        // Get the pixel as the desired point if we're in the image, otherwise
        // return 0.
        if self.in_bounds(x, y) {
            self.get_pixel(x, y).to_luma()[0].into()
        } else {
            0.
        }
    }
}
