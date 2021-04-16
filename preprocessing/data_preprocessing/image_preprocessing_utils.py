#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
from typing import Tuple

from PIL import Image
import numpy as np
import pandas as pd
from skimage import transform, exposure

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"



def load_image(path:str)-> np.ndarray:
    # TODO: write description
    with Image.open(path) as im:
        return np.ndarray(im)

def save_image(img:np.ndarray, path_to_output:str)->None:
    # TODO: write description
    img = Image.fromarray(img)
    img.save(path_to_output)

def resize_image(img:np.ndarray, new_size:Tuple[int, int])-> np.ndarray:
    # TODO: write description
    img=Image.fromarray(img)
    img=img.resize(new_size)
    return np.array(img)


def shear_image(img:np.ndarray, shear_factor:float)->np.ndarray:
    # TODO: write description
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=shear_factor)
    # Apply transform to image data
    modified_img = transform.warp(img, inverse_map=afine_tf)
    return modified_img

def rotate_image(img:np.ndarray, rotation_angle:int)->np.ndarray:
    # TODO: write description
    modified_image=transform.rotate(img, rotation_angle)
    return modified_image

def flip_image(img:np.ndarray, flip_type:str)->np.ndarray:
    # TODO: write description
    if not flip_type in ('horizontal','vertical'):
        raise AttributeError('Flip_type can be either \'horizontal\' or \'vertical\'. Got: %s' % flip_type)
    if flip_type == 'horizontal':
        modified_image=img[:,::-1]
    else:
        modified_image=img[::-1,:]
    return modified_image

def shift_image(img:np.ndarray, shift_factor:float)->np.ndarray:
    # TODO: write description
    transformation = transform.AffineTransform(translation=img)
    shifted = transform.warp(img, transformation, mode='wrap', preserve_range=True)
    shifted = shifted.astype(img.dtype)
    return shifted

def change_brightness(img:np.ndarray, brightness_factor:float)->np.ndarray:
    # TODO: write description
    modified_image = exposure.adjust_gamma(img, gamma=1+brightness_factor, gain=1)
    return modified_image

def zoom_image(img:np.ndarray, zoom_factor:float)->np.ndarray:
    pass