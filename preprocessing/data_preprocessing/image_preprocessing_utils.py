#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
from typing import Tuple

from PIL import Image
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import transform, exposure

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"



def load_image(path:str)-> np.ndarray:
    # TODO: write description
    with Image.open(path) as im:
        return np.array(im)

def load_batch_of_images(paths:Tuple[str,...])-> np.ndarray:
    # TODO: write description
    img=load_image(paths[0])
    img_shape=img.shape
    images_array=np.zeros(((len(paths),)+img_shape))
    images_array[0]=img
    for i in range(1, len(paths)):
        img=load_image(paths[i])
        images_array[i]=img
    return images_array.astype('uint8')


def save_image(img:np.ndarray, path_to_output:str)->None:
    # TODO: write description
    img = Image.fromarray(img)
    img.save(path_to_output)

def resize_image(img:np.ndarray, new_size:Tuple[int, int])-> np.ndarray:
    # TODO: write description
    img=Image.fromarray(img)
    img=img.resize(new_size)
    return np.array(img)

def show_image(img:np.ndarray)->None:
    Image.fromarray(img).show()


def shear_image(img:np.ndarray, shear_factor:float)->np.ndarray:
    # TODO: write description
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=shear_factor)
    # Apply transform to image data
    modified_img = transform.warp(img, inverse_map=afine_tf)
    return (modified_img*255.).astype('uint8')

def rotate_image(img:np.ndarray, rotation_angle:int)->np.ndarray:
    # TODO: write description
    modified_image=transform.rotate(img, rotation_angle)*255.
    return modified_image.astype('uint8')

def flip_image(img:np.ndarray, flip_type:str)->np.ndarray:
    # TODO: write description
    if not flip_type in ('horizontal','vertical'):
        raise AttributeError('Flip_type can be either \'horizontal\' or \'vertical\'. Got: %s' % flip_type)
    if flip_type == 'horizontal':
        modified_image=img[:,::-1]
    else:
        modified_image=img[::-1,:]
    return modified_image

def shift_image(img:np.ndarray, shift_vector:Tuple[float, float])->np.ndarray:
    # TODO: write description
    transformation = transform.AffineTransform(translation=shift_vector)
    shifted = transform.warp(img, transformation, mode='wrap', preserve_range=True)
    shifted = shifted.astype(img.dtype)
    return shifted

def change_brightness(img:np.ndarray, brightness_factor:float)->np.ndarray:
    # TODO: write description
    modified_image = exposure.adjust_gamma(img, gamma=1+brightness_factor, gain=1)
    return modified_image

def zoom_image(img:np.ndarray, zoom_factor:float)->np.ndarray:
    # TODO: write description
    result_image=ndimage.zoom(img, (zoom_factor, zoom_factor, 1))
    return result_image

def channel_random_noise(img:np.ndarray, num_channel:int, std:float)->np.ndarray:
    # TODO: write description
    image_size=img.shape[:2]
    noise=np.random.normal(scale=std,size=image_size)
    modified_image=img.copy().astype('float32')
    modified_image[:,:,num_channel]+=noise
    return modified_image.astype('uint8')

def crop_image(img:np.ndarray, bbox:Tuple[int,int,int,int])->np.ndarray:
    # TODO: write description
    x0,y0,x1,y1 = bbox
    if x0<0 or x1>img.shape[1] or y0<0 or y1>img.shape[0]:
        raise AttributeError("Some coordinates of bbox are negative or greater than "
                             "the image size. Provided bbox:%s"%bbox)
    return img[y0:y1, x0:x1]

def blur_image(img:np.ndarray, sigma:float=3)->np.ndarray:
    # TODO: write description
    modified_image=ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))
    return modified_image

def get_image_with_worse_quality(img:np.ndarray, rescaling_factor:float)->np.ndarray:
    # TODO: write description
    modified_image=transform.rescale(img, (rescaling_factor, rescaling_factor, 1), anti_aliasing=False)
    modified_image=transform.resize(modified_image, img.shape[:2])
    return (modified_image*255).astype('uint8')

def scale_img_to_0_1(img:np.ndarray)->np.ndarray:
    # TODO: write description
    modified_image=img/255.
    return modified_image


if __name__ == "__main__":
    img=load_image(r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\sorted_faces\1\1100142033_10.jpg')
    show_image(img)
    img=shear_image(img, -0.5)
    show_image(img)
