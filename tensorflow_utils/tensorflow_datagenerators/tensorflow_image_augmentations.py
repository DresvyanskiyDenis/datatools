#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TODO: write module description

List of functions:

    *

List of classes:

    *
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import Tuple, List

import tensorflow as tf

@tf.function
def rotate90_image(image):
    return tf.image.rot90(image)

@tf.function
def flip_vertical_image(image):
    return tf.image.flip_up_down(image)

@tf.function
def flip_horizontal_image(image):
    return tf.image.flip_left_right(image)


@tf.function
def crop_image(image, bbox):
    return tf.image.crop_to_bounding_box(image, *bbox)

@tf.function
def resize_image(image, new_size:List[int], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    return tf.image.resize(image, new_size, method)

@tf.function
def change_brightness_image(image, delta):
    return tf.image.adjust_brightness(image, delta)

@tf.function
def change_contrast_image(image, contrast_factor):
    return tf.image.adjust_contrast(image, contrast_factor)

@tf.function
def change_saturation_image(image, saturation_factor):
    return tf.image.adjust_saturation(image, saturation_factor)

@tf.function
def worse_quality_image(image, min_quality, max_quality):
    return tf.image.random_jpeg_quality(image, min_quality, max_quality)

@tf.function
def convert_to_grayscale_image(image):
    image=tf.image.rgb_to_grayscale(image)
    return tf.image.grayscale_to_rgb(image)

# The addition of the randomness to the augmentations

@tf.function
def random_rotate90_image(image, lbs):
    # use augmentation if probability is less than generated one
    return rotate90_image(image), lbs

@tf.function
def random_flip_vertical_image(image, lbs):
    return flip_vertical_image(image), lbs

@tf.function
def random_flip_horizontal_image(image, lbs):
    return flip_horizontal_image(image), lbs

@tf.function
def _crop_and_resize(image, crop_shape, result_shape):
    # crop randomly image and zoom it to original size
    result=tf.image.random_crop(image, crop_shape)
    return resize_image(result, result_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


@tf.function
def random_crop_image(image, lbs):
    old_shape = tf.shape(image)[:2]
    # create shape of cropped image
    crop_height = tf.cast(tf.shape(image)[0], dtype=tf.float32) * tf.random.uniform(shape=(),minval=0.7, maxval=0.9)
    crop_width = tf.cast(tf.shape(image)[1], dtype=tf.float32) * tf.random.uniform(shape=(), minval=0.7, maxval=0.9)
    crop_shape = (crop_height, crop_width, tf.cast(tf.shape(image)[-1], dtype=tf.float32))
    crop_shape = tf.cast(crop_shape, dtype=tf.int32)
    # crop image and resize it to the original size
    return _crop_and_resize(image, crop_shape, old_shape), lbs

@tf.function
def random_change_brightness_image(image, lbs, min_max_delta=0.35):
    delta=tf.random.uniform(shape=(), minval=-min_max_delta, maxval=min_max_delta)
    return change_brightness_image(image, delta), lbs

@tf.function
def random_change_contrast_image(image, lbs,  min_factor = 0.5, max_factor = 1.5):
    delta=tf.random.uniform(shape=(), minval=min_factor, maxval=max_factor)
    return change_contrast_image(image, delta), lbs

@tf.function
def random_change_saturation_image(image, lbs,  min_factor = 0.5, max_factor = 1.5):
    delta=tf.random.uniform(shape=(), minval=min_factor, maxval=max_factor)
    return change_saturation_image(image, delta), lbs

@tf.function
def random_worse_quality_image(image, lbs, min_factor = 25, max_factor = 99):
    return worse_quality_image(image, min_factor, max_factor), lbs

@tf.function
def random_convert_to_grayscale_image(image, lbs):
    return convert_to_grayscale_image(image), lbs
