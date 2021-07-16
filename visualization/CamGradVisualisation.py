#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize the gradients of neural network via CamGrad visualisation.

Module contains class, which performs GradCam visualisation for neural networks (implemented in Tensorflow)
CamGrad: https://arxiv.org/abs/1610.02391
Implementation of the functions taken from keras implementations: https://keras.io/examples/vision/grad_cam/

List of classes:
    * GradCAMVisualiser - Performs GradCam visualisation using static method perform_and_save_gradcam()
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import os
import re
from typing import Optional, Callable, Tuple

import numpy as np
import tensorflow as tf
from PIL.Image import Image
from matplotlib import cm

from preprocessing.data_normalizing_utils import VGGFace2_normalization
from preprocessing.data_preprocessing.image_preprocessing_utils import load_image
from tensorflow_utils.models.CNN_models import get_EMO_VGGFace2


class GradCAMVisualiser():
    """
        Methods:
            * _make_gradcam_heatmap - creates gradcam heatmap
            * _save_gradcam_image - saves created gradcam heatmap as image
            * perform_and_save_gradcam - static method, which combines two formet methods to perform and save gradcam heatmap
                and save it in provided directory.
    """

    def _make_gradcam_heatmap(self, img_array:np.ndarray, model:tf.keras.Model, layer_name:str,
                             pred_index:Optional[int]=None)->np.ndarray:
        # function is taken from https://keras.io/examples/vision/grad_cam/

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()



    def _save_gradcam_image(self, img_path:str, heatmap:np.ndarray, cam_path:str="cam.jpg", alpha:float=0.4)-> None:
        # function is partially taken from https://keras.io/examples/vision/grad_cam/

        # Load the original image
        img = load_image(img_path)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

    @staticmethod
    def perform_and_save_gradcam(img_path:str, path_to_save:str, model:tf.keras.Model,
                                 layer_name:str, class_index:Optional[int]=None, img_size:Tuple[int, int]=(224,224),
                                 preprocess_function:Optional[Callable[[np.ndarray], np.ndarray]]=None)->None:
        # create dir if not existed
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save, exist_ok=True)
        gradCAM_visualizer=GradCAMVisualiser()
        # load image
        img_filename=re.split(r'\\|/', img_path)[-1].split('.')[0]
        img=load_image(img_path)
        # resize
        img=np.array(Image.fromarray(img).resize(img_size))
        # preprocess if needed
        if preprocess_function is not None:
            img=preprocess_function(img)
        # make it in format needed to tensorflow.keras.Model
        img=img[np.newaxis,...]
        # compute heatmap
        heatmap=gradCAM_visualizer._make_gradcam_heatmap(img, model=model, layer_name=layer_name, pred_index=class_index)
        # save img with heatmap
        gradCAM_visualizer._save_gradcam_image(img_path, heatmap=heatmap,
                                               cam_path=os.path.join(path_to_save, '%s_gradCAM.jpg'%img_filename))


if __name__=='__main__':
    path_to_model_weights=r'C:\Users\Dresvyanskiy\Desktop\Projects\EMOVGGFace_model\weights_0_66_37_affectnet_cat.h5'
    img_path=r'C:\Users\Dresvyanskiy\Downloads\Telegram Desktop\choice_frame_attention_affecnet\AffWild2_v2\118\00293.jpg'
    layer_name='add_15'
    model=get_EMO_VGGFace2(path_to_model_weights)
    model.summary()
    GradCAMVisualiser.perform_and_save_gradcam(img_path, 'results', model, layer_name, preprocess_function=VGGFace2_normalization)
