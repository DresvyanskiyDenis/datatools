#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
from typing import Tuple, List, Optional, Union
import tensorflow as tf


def get_Dense_model(input_shape:Tuple[int,...],
                    dense_neurons: Tuple[int,...],
                    activations:Union[str,Tuple[str,...]]='relu',
                    dropout: Optional[float] = 0.3,
                    regularization:Optional[tf.keras.regularizers.Regularizer]=None,
                    output_neurons: Union[Tuple[int,...], int] = 7,
                    activation_function_for_output:str='softmax') -> tf.keras.Model:
    # TODO:write description
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


if __name__=="__main__":
    model=get_Dense_model(input_shape=(30,),
                    dense_neurons=(128,64,32),
                    activations='relu',
                    dropout= 0.3,
                    regularization=tf.keras.regularizers.l2(0.0001),
                    output_neurons = 4,
                    activation_function_for_output='softmax')
    model.summary()