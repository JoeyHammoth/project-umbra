#!/usr/bin/env python3
"""Computed Tomography Lab
==========================

Main executable for the computed tomography lab
"""

from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from attenuation_object import ObjectCollection, Rectangle, ImageObject
from misc import array_to_img
from projection import Projection
from scipy import ndimage
from scipy import signal

from PIL import Image


def plotting(dataset, datalabelarr, xlabel, ylabel, title, saveloc):
    """Plotting function is used to plot arrays. Mainly used to see how
    different filters and windows convolve.
    
    """
    for data, datalabel in zip(dataset, datalabelarr):
        plt.plot(data, label=datalabel, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(saveloc)
    plt.close()
    
def collection_fn(obj_type, image_loc="./input/tomatoes_a.png"):
    """Provides a choice of objects to be instantiated."""
    objects = ObjectCollection()
    if obj_type == 0:
        print("Squares")
        obj_1 = Rectangle(-1, 0, -1, 0, 1)
        obj_2 = Rectangle(0, 1, 0, 1, 1)
    elif obj_type == 1:
        print("Circles")
        obj_1 = Circle(0.0, 0.0, 2.0, 1.0)
        obj_2 = Circle(0.0, 0.0, 1.0, 1.0)
    else:
        print("Images")
        objects.append(ImageObject(Image.open(image_loc, mode='r')))

    if obj_type == 0 or obj_type == 1:
        objects.append(obj_1)
        objects.append(obj_2)
        
    return objects

def filter_fn(filter_type, xi_steps, x_step_size, high_pass_exp=1):
    """Provides a choice of filters to be used in the filtered back
    projection.
    
    """
    # High Pass
    if filter_type == 0:
        return np.abs(np.fft.fftfreq(xi_steps, x_step_size) ** high_pass_exp)
    # Low Pass
    elif filter_type == 1:
        return -np.abs(np.fft.fftfreq(xi_steps, x_step_size) + max(np.fft.fftfreq(xi_steps, x_step_size)))
    # None
    else:
        return 1

def window_fn(window_type, xi_steps, x_step_size, ham_alpha=10):
    """Provides a choice of windows to be used in the filtered back
    projection.
    
    """
    # Top Hat using sinc function
    if window_type == 0:
        return xi_steps * np.sinc((xi_steps * x_step_size) / 2)
    # General Hamming
    elif window_type == 1:
        return np.fft.fft(np.fft.fft(sp.signal.windows.general_hamming(xi_steps, ham_alpha)))
    # Kaiser
    elif window_type == 2:
        return np.fft.fft(np.fft.fft(sp.signal.windows.kaiser(xi_steps, 0)))
    # Flattop
    elif window_type == 3:
        return np.fft.fft(np.fft.fft(sp.signal.windows.flattop(xi_steps)))
    # Triangular
    elif window_type == 4:
        return np.fft.fft(np.fft.fft(sp.signal.windows.triang(xi_steps)))
    # Guassian Filter
    elif window_type == 5:
        return sp.ndimage.fourier_gaussian(np.absolute(np.fft.fftfreq(xi_steps, x_step_size)), 2)
    # None
    else:
        return 1

def main():
    """The main function in which everything you run should start."""
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()
        

    """ Instantating Objects """
    # Instantiating the objects for the projection using the 
    # collection function

    object_type = 0
    
    collection = collection_fn(object_type)

    # Checking if the image was imported correctly
    array_to_img(
        collection.to_array(np.linspace(-2, 2, 256), np.linspace(-2, 2, 256))
    ).save(output_dir / "original-image.png")


    # Get specific values of p_θ(ξ) to see if the projected 
    # attenuation function works
    for theta, xi in [(np.pi / 4, np.sqrt(2) / 2), (0.0, 0.5), (-np.pi, 0.0)]:
        projection = collection.project_attenuation(theta, xi, [-2.0, 2.0])
        print(
            "p at θ = {:.2}, ξ = {:.2} is: {:.2}".format(theta, xi, projection),
        )
        
    print("==> calculating projection")

    """ Creating Projection """
    # Instantiate a projection object and add the instantiated 
    # object to the projection. Also produces the top hat window
    # used for the filtered back projection. 

    image_projection = Projection([0.0, np.pi], 64, [-2.0, 2.0], 64)
    top_hat_w_arr = image_projection.add_object(collection, [-2.0, 2.0], 4)
    
    # Generate the sinogram of the object.
    array_to_img(image_projection.data).save(output_dir / "sinogram.png")

    """ Image Retrieval """
    # Generate back and filtered back projection of the image

    # Generate back projection
    array_to_img(image_projection.back_project()).save(output_dir / "back-projection.png")

    # Define variables used for calling filter and window functions
    xi_steps = image_projection.xi_steps
    xi_step_size = image_projection.xi_step_size

    # Generate filtered back projection with |k| filter and no window
    array_to_img(image_projection.filtered_back_project(filter_fn(0, xi_steps, xi_step_size))).save(output_dir / "filtered-back-projection.png")

    """ Noise Reduction """
    # Add noise to the projection and use a variety of filters and windows to reduce
    # the noise from the filtered back projection. 

    random_dist_type = 0

    # Find how the image changes as more noise is added using the chosen distribution
    for i in range(4):
        image_projection.add_noise(random_dist_type, 0.1)
        image_loc = "noise_images/"  + str(i) + "-filtered-back-projection.png"
        array_to_img(image_projection.filtered_back_project(filter_fn(0, xi_steps, xi_step_size))).save(output_dir / image_loc)

    # Generate the filtered back projection using an appropriate window and filter
    # to reduce the noise. 

    filter_type = 0
    window_type = 0

    # Using the top hat window generated directly when adding an object to the project 
    # using the formula in the notes
    filter_arr1 = filter_fn(filter_type, xi_steps, xi_step_size) * np.fft.fft(top_hat_w_arr)
    array_to_img(image_projection.filtered_back_project(filter_arr1)).save(output_dir / "tophat-noise-reduced-FBP.png")

    # Using a sinc function to represent a top hat window
    filter_arr2 = filter_fn(filter_type, xi_steps, xi_step_size) * window_fn(window_type, xi_steps, xi_step_size)
    array_to_img(image_projection.filtered_back_project(filter_arr2)).save(output_dir / "tophat-sinc-noise-reduced-FBP.png")


    """ Plotting """
    # Plotting the convolution of the filter and window

    data_arr = [filter_fn(filter_type, xi_steps, xi_step_size), window_fn(window_type, xi_steps, xi_step_size), filter_arr1[0]]
    name_arr = ['High Pass Filter', 'Top Hat Window', 'Convolution']
    plotting(data_arr, name_arr, 'sample', 'amplitude', 'Convolution of High Pass Filter and Top Hat Window', 'convolution-plot.png')

if __name__ == "__main__":
    main()