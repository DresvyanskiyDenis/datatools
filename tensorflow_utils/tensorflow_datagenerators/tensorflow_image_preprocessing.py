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


import tensorflow as tf



vgg_face2_mean = (91.4953, 103.8827, 131.0912)


@tf.function
def preprocess_image_VGGFace2(images:tf.Tensor, labels):
    images=tf.math.subtract(tf.cast(images[...,::-1], dtype=tf.float32), vgg_face2_mean)

    return images, labels

@tf.function
def preprocess_data_Xception(images:tf.Tensor, labels):
    images=tf.image.resize(images, [299, 299], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    images=tf.keras.applications.xception.preprocess_input(images)
    return images, labels

@tf.function
def preprocess_data_Xception(images:tf.Tensor, labels):
    images=tf.image.resize(images, [299, 299], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    images=tf.keras.applications.xception.preprocess_input(images)
    return images, labels

@tf.function
def preprocess_data_MobileNetv3(images:tf.Tensor, labels):
    images=tf.keras.applications.mobilenet_v3.preprocess_input(images)
    return images, labels
