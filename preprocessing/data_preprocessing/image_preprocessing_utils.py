#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains functions for video preprocessing.

List of functions:

    * TODO: list of function
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import Tuple

from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import transform, exposure


def load_image(path:str)-> np.ndarray:
    """Loads image using Pillow library.
        https://pillow.readthedocs.io/en/stable/

    :param path: str
            Path to the image to be loaded
    :return: np.ndarray
            Loaded image as 3D np.ndarray (height, width, channels)
    """
    with Image.open(path) as im:
        return np.array(im)

def load_batch_of_images(paths:Tuple[str,...])-> np.ndarray:
    """Loads batch of images, paths of which are in the passed Tuple.
       Returns it as  4D array (num_images, height, width, channels).
       It is assumed that all images have the same size.

    :param paths: Tuple[str,...]
            Paths to the images to be loaded.
    :return: np.ndarray
            Loaded images as 4D np.ndarray (num_images, height, width, channels)
    """
    img=load_image(paths[0])
    img_shape=img.shape
    images_array=np.zeros(((len(paths),)+img_shape))
    images_array[0]=img
    for i in range(1, len(paths)):
        img=load_image(paths[i])
        images_array[i]=img
    return images_array.astype('uint8')


def save_image(img:np.ndarray, path_to_output:str)->None:
    """Saves image using Pillow lib.

    :param img: np.ndarray
            Image as 3D np.ndarray
    :param path_to_output: str
            Path to the output
    :return: None
    """
    img = Image.fromarray(img)
    img.save(path_to_output)

def resize_image(img:np.ndarray, new_size:Tuple[int, int])-> np.ndarray:
    """Resizes image using Pillow lib.

    :param img: np.ndarray
            Image as 3D np.ndarray
    :param new_size: Tuple[int, int]
            The new size of the image
    :return: np.ndarray
            Resized image as 3D np.ndarray (height, width, channels)
    """
    img=Image.fromarray(img)
    img=img.resize(new_size)
    return np.array(img)

def show_image(img:np.ndarray)->None:
    """Shows image using Pillow lib.

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :return: None
    """
    Image.fromarray(img).show()


def shear_image(img:np.ndarray, shear_factor:float)->np.ndarray:
    """Shears image.
    Skicit-image is used: https://scikit-image.org/

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param shear_factor: float
            Shear angle in counter-clockwise direction as radians.
    :return: np.ndarray
            Sheared image as 3D np.ndarray
    """
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=shear_factor)
    # Apply transform to image data
    modified_img = transform.warp(img, inverse_map=afine_tf)
    return (modified_img*255.).astype('uint8')

def rotate_image(img:np.ndarray, rotation_angle:int)->np.ndarray:
    """Rotates image.
    Skicit-image is used: https://scikit-image.org/

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param rotation_angle: int
            rotation angle in degrees.
    :return: np.ndarray
            Rotated image as 3D np.ndarray
    """
    modified_image=transform.rotate(img, rotation_angle)*255.
    return modified_image.astype('uint8')

def flip_image(img:np.ndarray, flip_type:str)->np.ndarray:
    """Flips image (horizontally or vertically).

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param flip_type: str
            Either horizontal or vertical options are supported.
    :return: np.ndarray
            Flipped image as 3D np.ndarray
    """
    if not flip_type in ('horizontal','vertical'):
        raise AttributeError('Flip_type can be either \'horizontal\' or \'vertical\'. Got: %s' % flip_type)
    if flip_type == 'horizontal':
        modified_image=img[:,::-1]
    else:
        modified_image=img[::-1,:]
    return modified_image

def shift_image(img:np.ndarray, shift_vector:Tuple[float, float])->np.ndarray:
    """Shifts image on certain distance.

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param shift_vector: Tuple[float, float]
            The translation vector, the values should be defined in the amount of pixels
            (on how much pixels the image should be shifted).
    :return: np.ndarray
            Shifted image as 3D np.ndarray
    """
    transformation = transform.AffineTransform(translation=shift_vector)
    shifted = transform.warp(img, transformation, mode='wrap', preserve_range=True)
    shifted = shifted.astype(img.dtype)
    return shifted

def change_brightness(img:np.ndarray, brightness_factor:float)->np.ndarray:
    """Changes the brightness of the image.

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param brightness_factor: float
            Non-negative float number. (>1 the image becomes darker than the input image, 1< becomes lighter)
    :return: np.ndarray
            Changed image as 3D np.ndarray
    """
    modified_image = exposure.adjust_gamma(img, gamma=1+brightness_factor, gain=1)
    return modified_image

def zoom_image(img:np.ndarray, zoom_factor:float)->np.ndarray:
    """Zooms image by provided factor. Basically, this function makes image bigger or smaller.

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param zoom_factor: float
            The factor by which image will be zoomed. For example, zoom_factor=2 makes image twice bigger.
    :return: np.ndarray
            Zoomed image as 3D np.ndarray
    """
    result_image=ndimage.zoom(img, (zoom_factor, zoom_factor, 1))
    return result_image

def channel_random_noise(img:np.ndarray, num_channel:int, std:float)->np.ndarray:
    """Adds random gaussian noise (with defined std) to the chosen color channel.

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param num_channel: int
            The number of color channel to which the noise will be added.
    :param std: float
            The standard deviation parameter for random gaussian noise.
    :return: np.ndarray
            Changed image as 3D np.ndarray
    """
    image_size=img.shape[:2]
    noise=np.random.normal(scale=std,size=image_size)
    modified_image=img.copy().astype('float32')
    modified_image[:,:,num_channel]+=noise
    return modified_image.astype('uint8')

def crop_image(img:np.ndarray, bbox:Tuple[int,int,int,int])->np.ndarray:
    """Crops provided bounding box from the image, making area black.

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param bbox: Tuple[int,int,int,int]
            The area, which should be cut off from image (by coloring it in black).
    :return: np.ndarray
            Changed image as 3D np.ndarray
    """
    x0,y0,x1,y1 = bbox
    if x0<0 or x1>img.shape[1] or y0<0 or y1>img.shape[0]:
        raise AttributeError("Some coordinates of bbox are negative or greater than "
                             "the image size. Provided bbox:%s"%bbox)
    return img[y0:y1, x0:x1]

def blur_image(img:np.ndarray, sigma:float=3)->np.ndarray:
    """Blues image. Scipy are used.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param sigma: float
            The standard deviation parameter for gaussian kernel. (good results for [1,5]).
    :return: np.ndarray
            Changed image as 3D np.ndarray
    """
    modified_image=ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))
    return modified_image

def get_image_with_worse_quality(img:np.ndarray, rescaling_factor:float)->np.ndarray:
    """Worsens image quality by some factor. It is done by firstly scaling image down (decreasing height and width)
       and then up (increasing back to the first values).

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :param rescaling_factor: float
            The factor, on which the image will be worsen. The value should be from 0 to 1 (excluding).
    :return: np.ndarray
            Changed image as 3D np.ndarray
    """
    modified_image=transform.rescale(img, (rescaling_factor, rescaling_factor, 1), anti_aliasing=False)
    modified_image=transform.resize(modified_image, img.shape[:2])
    return (modified_image*255).astype('uint8')

def scale_img_to_0_1(img:np.ndarray)->np.ndarray:
    """Rescales pixel values to the [0,1].

    :param img: np.ndarray
            Image as 3D np.ndarray.
    :return: np.ndarray
            Changed image as 3D np.ndarray
    """
    modified_image=img/255.
    return modified_image


if __name__ == "__main__":
    img=load_image(r'C:\Users\Professional\Desktop\Article_pictures\AffWild2\Angry_1.jpg')
    show_image(img)
    print(img.shape)
    img=get_image_with_worse_quality(img, 10)
    show_image(img)
    print(img.shape)
