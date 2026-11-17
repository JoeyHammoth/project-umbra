//! The projection information, pϑ(ξ) is, at its core, a function ℝ² → ℝ.
//! Within the computer, this will be stored as ℕ² → ℝ, where ℝ is approximated
//! by a float.
//!
//! Instead of defining a whole lot of different functions to handle the 2D
//! array (and then needing to remember the number of steps in theta, or the
//! range of xi values), we use a struct to store everything together.

use crate::object::AttenuationObject;
// use ndarray::;
use ndarray::{prelude::*, Zip};
use num::Complex;
use rustfft;

/// The main struct that stores all the information.
///
/// Internally, the data is stored in a 2D array, with the size of the array
/// determining the number of steps in theta and xi that are used.  The range in
/// theta and xi is also stored within the struct and all functions that work on
/// projections can be implemented as methods.
#[allow(dead_code)]
pub struct Projection {
    data: Array2<f64>,
    theta_min: f64,
    theta_max: f64,
    theta_steps: usize,
    theta_step_size: f64,
    xi_min: f64,
    xi_max: f64,
    xi_steps: usize,
    xi_step_size: f64,
}

impl Projection {
    /// Create a new empty projection object.  The range of theta values and
    /// corresponding number of subdivisions are specified in `(theta_min,
    /// theta_max)` and `theta_steps`.  Similarly, the range of xi values and
    /// corresponding number of subdivisions are specified in `(xi_min, xi_max)`
    /// and `xi_steps`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    ///
    /// let projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// ```
    pub fn new(
        (theta_min, theta_max): (f64, f64),
        theta_steps: usize,
        (xi_min, xi_max): (f64, f64),
        xi_steps: usize,
    ) -> Projection {
        // Make sure that theta_min and theta_max are correctly ordered, and
        // calculate the step size.  Then repeat with xi.
        let (theta_min, theta_max) = if theta_min < theta_max {
            (theta_min, theta_max)
        } else {
            (theta_max, theta_min)
        };
        let theta_step_size = (theta_max - theta_min) / (theta_steps as f64);
        let (xi_min, xi_max) = if xi_min < xi_max {
            (xi_min, xi_max)
        } else {
            (xi_max, xi_min)
        };
        let xi_step_size = (xi_max - xi_min) / (xi_steps as f64);

        Projection {
            data: Array2::zeros((theta_steps, xi_steps)),
            theta_min,
            theta_max,
            theta_steps,
            theta_step_size,
            xi_min,
            xi_max,
            xi_steps,
            xi_step_size,
        }
    }

    /// Return the value of theta corresponding to a particular index.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    ///
    /// let projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// assert_eq!(projection.theta(2), 2.0 * PI / 128.0);
    /// ```
    #[inline]
    pub fn theta(&self, idx: usize) -> f64 {
        self.theta_min + (idx as f64) * self.theta_step_size
    }

    /// Return xi at a particular step index.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    ///
    /// let mut projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// assert_eq!(projection.xi(2), -2.0 + 2.0 * 4.0 / 128.0);
    /// ```
    #[inline]
    pub fn xi(&self, idx: usize) -> f64 {
        self.xi_min + (idx as f64) * self.xi_step_size
    }

    /// Return the value of the projection at any arbitrary point using
    /// interpolation (for a given value of theta).  This allows for the
    /// projection to be defined for all values in the range of xi values used
    /// in the projection.
    ///
    /// For values of xi which are outside of the range in the projection, the
    /// edge point is returned:
    ///
    /// ```text
    ///        |             XXX           |
    ///        |            X   X          |
    /// XXXXXXXX           X     X         |
    ///        |X XX       X     X         |
    ///        | X  X     X       X        |
    ///        |     X   X         X       |
    ///        |      XXX           X      XXXXXXXX
    ///        |                    X    XX|
    ///        |                     X  X  |
    ///        |                      XX   |
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    ///
    /// let mut projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// assert_eq!(projection.get_continuous(4, -2.0), 0.0);
    /// assert_eq!(projection.get_continuous(4, 0.0), 0.0);
    /// assert_eq!(projection.get_continuous(4, 1e3), 0.0);
    /// ```
    #[inline]
    pub fn get_continuous(&self, theta_idx: usize, xi: f64) -> f64 {
        self.get_continuous_custom(theta_idx, xi, &self.data)
    }

    /// This functions behaves exactly like `get_continuous`, but instead of used the
    /// data array contained within the projection, an other array can be
    /// specified (which must be of the same shape).
    ///
    /// This is intended to be used internally when manipulating data that has
    /// been filtered Fourier transformed.
    #[inline]
    fn get_continuous_custom(&self, theta_idx: usize, xi: f64, data: &Array2<f64>) -> f64 {
        let xi_idx = (xi - self.xi_min) / self.xi_step_size;
        if xi_idx < 0.0 {
            data[[theta_idx, 0]]
        } else if xi_idx >= self.xi_steps as f64 {
            data[[theta_idx, self.xi_steps - 1]]
        } else {
            data[[theta_idx, xi_idx as usize]]
        }
    }

    /// Return a reference to the underlying data array.  The data is a 2D array
    /// with the first index corresponding to a particular value of theta, and
    /// the second index corresponds to xi.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    ///
    /// let projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// for row in projection.data().genrows() {
    ///     for &entry in row {
    ///         assert_eq!(entry, 0.0);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn data(&self) -> &Array2<f64> {
        &self.data
    }

    /// Return a mutable reference to the underlying data array.  The data is a
    /// 2D array with the first index corresponding to a particular value of
    /// theta, and the second index corresponds to xi.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    ///
    /// let mut projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// for row in projection.data_mut().genrows_mut() {
    ///     for entry in row {
    ///         *entry += 1.0;
    ///     }
    /// }
    /// for row in projection.data().genrows() {
    ///     for &entry in row {
    ///         assert_eq!(entry, 1.0);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn data_mut(&mut self) -> &mut Array2<f64> {
        &mut self.data
    }

    /// Add an object to the projection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    /// use ctlab::object::Circle;
    ///
    /// let circle = Circle::new(1.0, 1.0, 3.0, 10.0);
    /// let mut projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// projection.add_object(&circle, (-5.0, 5.0));
    /// ```
    pub fn add_object<T>(&mut self, obj: &T, eta_range: (f64, f64))
    where
        T: AttenuationObject,
        T: Sync,
    {
        unimplemented!();
    }

    /// Reconstruct the original image by back-projection.
    ///
    /// The resulting array size is square and has the same width as the number
    /// of xi steps in the original projection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    /// use ctlab::object::Circle;
    ///
    /// let circle = Circle::new(1.0, 1.0, 3.0, 10.0);
    /// let mut projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    /// projection.add_object(&circle, (-5.0, 5.0));
    /// let bp = projection.back_project();
    /// ```
    pub fn back_project(&self) -> Array2<f64> {
        unimplemented!();
    }

    /// Reconstruct the original image by filtered back-projection.
    ///
    /// The resulting array size is square and has the same width as the number
    /// of xi steps in the original projection.
    ///
    /// The filter is an arbitrary function on floats and acts on the
    /// wavenumber.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate ctlab;
    /// extern crate num;
    ///
    /// use std::f64::consts::PI;
    /// use ctlab::projection::Projection;
    /// use ctlab::object::Circle;
    /// use num::Complex;
    ///
    /// fn main() {
    ///     let circle = Circle::new(1.0, 1.0, 3.0, 10.0);
    ///     let mut projection = Projection::new((0.0, PI), 128, (-2.0, 2.0), 128);
    ///     projection.add_object(&circle, (-5.0, 5.0));
    ///     let bp = projection.filtered_back_project(|k| Complex::new(k.abs(), 0.0));
    /// }
    /// ```
    pub fn filtered_back_project<F>(&self, f: F) -> Array2<f64>
    where
        F: Fn(f64) -> Complex<f64>,
    {
        // The FFT need to pre-compute a few things before being used.  We
        // initialize them here.  The next two lines are commented out for now
        // because they need to be used in order to compile.
        // let forward = rustfft::FFTplanner::new(false).plan_fft(self.xi_steps);
        // let backward = rustfft::FFTplanner::new(true).plan_fft(self.xi_steps);

        // The FFT also needs to use complex numbers, and needs separate input
        // and output arrays to read from/write to.  As a result, we'll copy the
        // data we currently have into a new array of complex numbers.
        let mut data: Array2<Complex<f64>> = self.data.mapv(|v| Complex::new(v, 0.0));
        let mut data_ft: Array2<Complex<f64>> = Array2::zeros(self.data.dim());

        // To use the FFT, the general form is:
        //
        // ```
        // Zip::from(data.genrows_mut())
        //     .and(data_ft.genrows_mut())
        //     .par_apply(|mut r, mut r_ft| {
        //         forward.process(r.as_slice_mut().unwrap(), r_ft.as_slice_mut().unwrap())
        //     });
        // ```
        //
        // The first two lines iterate of the rows of the `data` and `data_ft`
        // arrays.  We then apply in parallel the forward FFT, converting the
        // input row into the output row.  After this, data_ft will contain the
        // transformed data.
        //
        // The reverse process is very similar.

        unimplemented!();
    }

}
