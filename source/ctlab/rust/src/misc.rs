//! Various functions which do not fit anywhere else in particular and might be helpful.

use image::GrayImage;
use ndarray::prelude::*;

/// Converts the array into an image, mapping the maximum value within the array
/// to white and 0 to black.
///
/// # Examples
///
/// ```
/// extern crate ctlab;
/// extern crate ndarray;
/// use ctlab::misc::array_to_img;
/// use ndarray::Array2;
///
/// fn main() {
///     let mut a = Array2::zeros((256, 256));
///     let mut val = 1.;
///     for entry in a.diag_mut() {
///         *entry = val;
///         val += 1.;
///     }
///
///     let img = array_to_img(&a);
///     img.save("/tmp/test-max.png").unwrap();
/// }
/// ```
pub fn array_to_img(d: &Array2<f64>) -> GrayImage {
    let &max = d
        .iter()
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    array_to_img_with_threshold(&d.map(|v| (v / max)), 1.)
}

/// Converts the array into an image, mapping the values at the threshold to
/// white and 0 to black.  Any values below 0 are mapped to black and values
/// above the threshold are mapped to black.  Any value in the array which is
/// not finite is mapped to black.
///
/// Unlike `array_to_img_max`, this function *does not* adjust the brightness
/// curve and uses a simple linear mapping from `f64` to `u8` brightness values.
///
/// # Examples
///
/// ```
/// extern crate ctlab;
/// extern crate ndarray;
/// use ctlab::misc::array_to_img_with_threshold;
/// use ndarray::Array2;
///
/// fn main() {
///     let mut a = Array2::zeros((256, 256));
///     let mut val = 1.;
///     for entry in a.diag_mut() {
///         *entry = val;
///         val += 1.;
///     }
///
///     let img = array_to_img_with_threshold(&a, 128.);
///     img.save("/tmp/test-threshold.png").unwrap();
/// }
/// ```
pub fn array_to_img_with_threshold(d: &Array2<f64>, threshold: f64) -> GrayImage {
    let mut img = GrayImage::new(d.shape()[0] as u32, d.shape()[1] as u32);

    for (xi, yi, p) in img.enumerate_pixels_mut() {
        let new_p = d[[xi as usize, yi as usize]] / threshold * 255.;
        p[0] = if new_p < 0. || !new_p.is_finite() {
            0
        } else if new_p >= 255. {
            255
        } else {
            new_p as u8
        }
    }

    img
}
