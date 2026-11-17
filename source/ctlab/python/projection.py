"""Projection
=============

The projection information, pϑ(ξ) is, at its core, a function ℝ² → ℝ.  Within
the computer, this will be stored as ℕ² → ℝ, where ℝ is approximated by a
float.

Instead of defining a whole lot of different functions to handle the 2D array
(and then needing to remember the number of steps in theta, or the range of xi
values), we use a struct to store everything together.

"""

from typing import Tuple, Callable
from attenuation_object import AttenuationObject

import numpy as np


class Projection:
    """The class to store all the information pertaining to a particular
    projection.

    Internally, the data is stored in a 2D array, with the size of the array
    determining the number of steps in theta and xi that are used.  The range
    in theta and xi is also stored within the struct and all functions that
    work on projections can be implemented as methods.

    """

    def __init__(
        self,
        theta_range: Tuple[float, float],
        theta_steps: int,
        xi_range: Tuple[float, float],
        xi_steps: int,
    ):
        """Create a new empty projection object.

        The range of theta values and corresponding number of subdivisions are
        specified in `(theta_min, theta_max)` and `theta_steps`.  Similarly,
        the range of xi values and corresponding number of subdivisions are
        specified in `(xi_min, xi_max)` and `xi_steps`.

        """
        # Make sure that theta_min and theta_max are correctly ordered, and
        # calculate the step size.  Then repeat with xi.
        self.theta_min = min(theta_range)
        self.theta_max = max(theta_range)
        self.theta_steps = theta_steps
        self.theta_step_size = (self.theta_max - self.theta_min) / theta_steps

        self.xi_min = min(xi_range)
        self.xi_max = max(xi_range)
        self.xi_steps = xi_steps
        self.xi_step_size = (self.xi_max - self.xi_min) / xi_steps

        self.data = np.zeros((theta_steps, xi_steps), dtype=np.float64)

    def theta(self, idx: int) -> float:
        """Return the value of theta corresponding to a particular index."""
        return self.theta_min + idx * self.theta_step_size

    def xi(self, idx: int) -> float:
        """Return xi at a particular step index."""
        return self.xi_min + idx * self.xi_step_size

    def get_continuous(self, theta_idx: int, xi: float) -> float:
        """Return the value of the projection at any arbitrary point using
        interpolation (for a given value of theta).

        This allows for the projection to be defined for all values in the
        range of xi values used in the projection. For values of xi which are
        outside of the range in the projection, the edge point is returned:

        ```text
            |             XXX           |
            |            X   X          |
        XXXXX           X     X         |
            |X XX       X     X         |
            | X  X     X       X        |
            |     X   X         X       |
            |      XXX           X      XXXXXXXX
            |                    X    XX|
            |                     X  X  |
            |                      XX   |
        ```

        """
        return self.get_continuous_custom(theta_idx, xi, self.data)

    def get_continuous_custom(self, theta_idx: int, xi: float, data) -> float:
        """This functions behaves exactly like `get_continuous`, but instead of
        used the data array contained within the projection, an other array can
        be specified (which must be of the same shape).

        This is intended to be used internally when manipulating data that has
        been filtered Fourier transformed.

        """
        xi_idx = (xi - self.xi_min) / self.xi_step_size
        if xi_idx < 0:
            return data[theta_idx, 0]
        if xi_idx >= self.xi_steps:
            return data[theta_idx, -1]
        return data[theta_idx, int(xi_idx)]        

    def add_object(self, obj: AttenuationObject, eta_range: Tuple[float, float], L):
        """Add an object to the projection and computes the top hat function array
        used for the filtered back projection.
        
        """
        
        # Loop through the 2D data array and for each element, call the project_attenuation 
        # function from the attenuation object to compute the projected attenuation for 
        # a specific theta and xi value. 
        top_hat_w_arr = np.zeros((len(self.data), len(self.data[0])))
        for theta_index in range(len(self.data)):
            for xi_index in range(len(self.data[theta_index])):
                self.data[theta_index][xi_index] = obj.project_attenuation(self.theta(theta_index), self.xi(xi_index), eta_range)
                if self.xi(xi_index) > -(L/2) and self.xi(xi_index) < (L/2):
                    top_hat_w_arr[theta_index][xi_index] = 1/L
        return top_hat_w_arr
                
    def add_noise(self, distribution, noise_factor, location=0, stdev=1):
        """Using random samples from a different distributions to generate a 2D array that is of the same 
        size with the data array and add this array to the data array as a means of adding noise to the
        porjection. 
        
        """
        # Gaussian Distribution
        if distribution == 0:
            noise_arr = np.random.normal(loc=location, scale=stdev, size=(len(self.data), len(self.data[0]))) * noise_factor
        # Poisson Distribution
        elif distribution == 1:
            noise_arr = np.random.poisson(size=(len(self.data), len(self.data[0]))) * noise_factor
        # Logistic Distribution
        else:
            noise_arr = np.random.logistic(loc=location, scale=stdev, size=(len(self.data), len(self.data[0]))) * noise_factor
        self.data = np.add(self.data, noise_arr)
    
    def back_project(self):
        """Reconstruct the original image by back-projection.

        The resulting array size is square and has the same width as the number
        of xi steps in the original projection.

        """
        back_proj_arr = np.zeros((self.xi_steps, self.xi_steps))
        for x in range(self.xi_steps):
            for y in range(self.xi_steps):
                theta_arr = []
                for theta in range(self.theta_steps):
                    theta_arr.append(self.get_continuous(theta, ((np.cos(self.theta(theta)) * self.xi(x)) + (np.sin(self.theta(theta)) * self.xi(y)))))
                back_proj_arr[x][y] = np.sum(theta_arr)
                
        return back_proj_arr
        
        #raise NotImplementedError()
    def filtered_back_project(self, f: Callable[[float], float]):
        """Reconstruct the original image by filtered back-projection.

        The resulting array size is square and has the same width as the number
        of xi steps in the original projection.

        The filter is an arbitrary function on floats and acts on the
        wavenumber.

        """
        ft_arr = np.zeros((len(self.data), len(self.data[0])))
        ft_proj_att = np.fft.fft(self.data)
        for row in range(len(self.data)):
            for col in range(len(self.data[row])):
                if (f.ndim == 1):
                    ft_arr[row][col] = ft_proj_att[row][col] * f[col]
                else:
                    ft_arr[row][col] = ft_proj_att[row][col] * f[row][col]
        ift_arr = np.fft.ifft(ft_arr) 

        back_proj_arr = np.zeros((self.xi_steps, self.xi_steps))
        for x in range(self.xi_steps):
            for y in range(self.xi_steps):
                theta_arr = []
                for theta in range(self.theta_steps):
                    theta_arr.append(self.get_continuous_custom(theta, ((np.cos(self.theta(theta)) * self.xi(x)) + (np.sin(self.theta(theta)) * self.xi(y))), ift_arr))
                back_proj_arr[x][y] = np.sum(theta_arr)
                
        return back_proj_arr 

        
        #raise NotImplementedError()
