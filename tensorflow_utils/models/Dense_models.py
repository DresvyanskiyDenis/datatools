#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the functions for building the tf.keras Dense models, without RNN and CNN lazers.

List of functions:
    * get_Dense_model - creates a Dense tf.keras Model according to the specified parameters.
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import Tuple, Optional, Union
import tensorflow as tf

def get_Dense_model(input_shape:Tuple[int,...],
                    dense_neurons: Tuple[int,...],
                    activations:Union[str,Tuple[str,...]]='relu',
                    dropout: Optional[float] = 0.3,
                    regularization:Optional[tf.keras.regularizers.Regularizer]=None,
                    output_neurons: Union[Tuple[int,...], int] = 7,
                    activation_function_for_output:str='softmax') -> tf.keras.Model:
    """Creates the tf.keras model with deliberate number of layers.

    :param input_shape: Tuple[int,...]
            The input shape of the model. Should be a Tuple with integer values.
    :param dense_neurons: Tuple[int,...]
            The number of neurons on each consecutive layers. Should be a Tuple with integer values.
    :param activations: Union[str,Tuple[str,...]]
            The activation functions of each layer presented in dense_neurons. If str, applies chosen activation
            to all layers except of the last layer.
    :param dropout: Optional[float]
            If indicated, applies the dropout to every layer except of the final.
    :param regularization: Optional[tf.keras.regularizers.Regularizer]
            If indicated, applies the regularization to every layer except of the final.
    :param output_neurons: Union[Tuple[int,...], int]
            The number of the output neurons (in the last layer). If Tuple[int,...], creates several output layers with
            the number of neurons presented in Tuple.
    :param activation_function_for_output: str
            The activation function for the output layer (layers).
    :return: tf.keras.Model
            The dense model created according to the specified parameters.
    """
    input_layer=tf.keras.layers.Input(input_shape)
    # create first Dense layer
    if isinstance(activations, tuple): activation=activations[0]
    else: activation=activations
    x = tf.keras.layers.Dense(dense_neurons[0], kernel_regularizer=regularization, activation=activation)(input_layer)
    # iterate throug dense_neurons tuple and construct Dense layers accordingly
    for layer_idx in range(1,len(dense_neurons)):
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
        if isinstance(activations, tuple): activation = activations[layer_idx]
        else: activation = activations
        x = tf.keras.layers.Dense(dense_neurons[layer_idx], kernel_regularizer=regularization, activation=activation)(x)
    # last layer
    # If model should have several output layers, then create several layers, otherwise one
    if isinstance(output_neurons, tuple):
        output_layers = []
        for num_output_neurons in output_neurons:
            output_layer_i = tf.keras.layers.Dense(num_output_neurons, activation=activation_function_for_output)(x)
            output_layers.append(output_layer_i)
    else:
        output_layers = tf.keras.layers.Dense(output_neurons, activation=activation_function_for_output)(x)
        # when using tf.keras.Model, it should be always a list (even when it has only 1 element)
        output_layers = [output_layers]
    model=tf.keras.Model(inputs=[input_layer], outputs=output_layers)
    return model
