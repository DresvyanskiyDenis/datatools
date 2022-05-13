#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
from typing import Tuple, List, Optional, Union
import tensorflow as tf

from tensorflow_utils.models.resnet50 import resnet50_backend


def get_mobilenet_v2_model(input_shape: Tuple[int, int, int],
                           dense_neurons_after_conv: List[int],
                           dropout: float = 0.3,
                           output_neurons: int = 7, pooling_at_the_end: Optional[str] = None,
                           pretrained: bool = True) -> tf.keras.Model:
    if pretrained:
        weights = 'imagenet'
    else:
        weights = None
    mobilenet_v2_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                                        include_top=False,
                                                                        weights=weights,
                                                                        pooling=pooling_at_the_end)
    x = mobilenet_v2_model.output
    for idx_dense_layer_neurons in range(len(dense_neurons_after_conv) - 1):
        neurons = dense_neurons_after_conv[idx_dense_layer_neurons]
        x = tf.keras.layers.Dense(neurons, activation='relu')(x)
        if dropout: x = tf.keras.layers.Dropout(dropout)(x)
    # last layer
    x = tf.keras.layers.Dense(dense_neurons_after_conv[-1], activation='relu')(x)
    output = tf.keras.layers.Dense(output_neurons, activation='softmax')(x)
    result_model = tf.keras.Model(inputs=mobilenet_v2_model.inputs, outputs=[output])
    return result_model


def _get_pretrained_VGGFace2_model(path_to_weights: str, pretrained: bool = True) -> tf.keras.Model:
    # inputs are of size 224 x 224 x 3
    input_dim = (224, 224, 3)
    inputs = tf.keras.layers.Input(shape=input_dim, name='base_input')
    x = resnet50_backend(inputs)

    # AvgPooling
    x = tf.keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='dim_proj')(x)
    y = tf.keras.layers.Dense(8631, activation='softmax',
                              use_bias=False, name='classifier_low_dim')(x)

    # Compile
    model = tf.keras.models.Model(inputs=inputs, outputs=y)
    model.compile(optimizer='SGD', loss='mse')
    if pretrained:
        model.load_weights(path_to_weights)
    return model


def get_modified_VGGFace2_resnet_model(dense_neurons_after_conv: Tuple[int,...],
                                       dropout: float = 0.3,
                                       regularization:Optional[tf.keras.regularizers.Regularizer]=None,
                                       output_neurons: Union[Tuple[int,...], int] = 7, pooling_at_the_end: Optional[str] = None,
                                       pretrained: bool = True,
                                       path_to_weights: Optional[str] = None) -> tf.keras.Model:
    pretrained_VGGFace2 = _get_pretrained_VGGFace2_model(path_to_weights, pretrained=pretrained)
    x=pretrained_VGGFace2.get_layer('activation_48').output
    # take pooling or not
    if pooling_at_the_end is not None:
        if pooling_at_the_end=='avg':
            x=tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling_at_the_end=='max':
            x=tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise AttributeError('Parameter pooling_at_the_end can be either \'avg\' or \'max\'. Got %s.'%(pooling_at_the_end))
    # create Dense layers
    for dense_layer_idx in range(len(dense_neurons_after_conv)-1):
        num_neurons_on_layer=dense_neurons_after_conv[dense_layer_idx]
        x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu', kernel_regularizer=regularization)(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    # pre-last Dense layer
    num_neurons_on_layer=dense_neurons_after_conv[-1]
    x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu')(x)
    # If outputs should be several, then create several layers, otherwise one
    if isinstance(output_neurons, tuple):
        output_layers=[]
        for num_output_neurons in output_neurons:
            if dropout:
                output_layer_i = tf.keras.layers.Dropout(dropout)(x)
            output_layer_i = tf.keras.layers.Dense(128, activation='relu')(output_layer_i)
            output_layer_i=tf.keras.layers.Dense(num_output_neurons, activation='softmax')(output_layer_i)
            #output_layer_i=tf.keras.layers.Reshape((-1, 1))(output_layer_i)
            output_layers.append(output_layer_i)
    else:
        output_layers = tf.keras.layers.Dense(output_neurons, activation='softmax')(x)
        # in tf.keras.Model it should be always a list (even when it has only 1 element)
        output_layers = [output_layers]
    # create model
    model=tf.keras.Model(inputs=pretrained_VGGFace2.inputs, outputs=output_layers)
    del pretrained_VGGFace2
    return model


def get_EMO_VGGFace2(path:str)->tf.keras.Model:
    # construct old model
    pretrained_VGGFace2 = _get_pretrained_VGGFace2_model("", pretrained=False)
    x = pretrained_VGGFace2.get_layer('activation_48').output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=512, activation='relu', name='embeddings_layer')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=7, activation='softmax')(x)
    EMO_VGGFace2_model = tf.keras.Model(inputs=pretrained_VGGFace2.inputs, outputs=[x])
    EMO_VGGFace2_model.load_weights(path)
    del pretrained_VGGFace2
    tf.keras.backend.clear_session()
    return EMO_VGGFace2_model

def get_pretrained_modified_EMO_VGGFace2(dense_neurons_after_conv: Tuple[int,...],
                                       dropout: float = 0.3,
                                       regularization:Optional[tf.keras.regularizers.Regularizer]=None,
                                       output_neurons: Union[Tuple[int,...], int] = 7, pooling_at_the_end: Optional[str] = None,
                                       path_to_weights: Optional[str] = None) -> tf.keras.Model:
    # construct old model
    EMO_VGGFace2_model=get_EMO_VGGFace2(path_to_weights)
    # start to construct new model by cutting off old one
    x = EMO_VGGFace2_model.get_layer('activation_48').output
    # take pooling or not
    if pooling_at_the_end is not None:
        if pooling_at_the_end == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling_at_the_end == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise AttributeError(
                'Parameter pooling_at_the_end can be either \'avg\' or \'max\'. Got %s.' % (pooling_at_the_end))
    # create Dense layers
    for dense_layer_idx in range(len(dense_neurons_after_conv) - 1):
        num_neurons_on_layer = dense_neurons_after_conv[dense_layer_idx]
        x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu', kernel_regularizer=regularization)(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    # pre-last Dense layer
    num_neurons_on_layer = dense_neurons_after_conv[-1]
    x = tf.keras.layers.Dense(num_neurons_on_layer, activation='relu')(x)
    # If outputs should be several, then create several layers, otherwise one
    if isinstance(output_neurons, tuple):
        output_layers = []
        for num_output_neurons in output_neurons:
            output_layer_i = tf.keras.layers.Dense(num_output_neurons, activation='softmax')(x)
            # output_layer_i=tf.keras.layers.Reshape((-1, 1))(output_layer_i)
            output_layers.append(output_layer_i)
    else:
        output_layers = tf.keras.layers.Dense(output_neurons, activation='softmax')(x)
        # in tf.keras.Model it should be always a list (even when it has only 1 element)
        output_layers = [output_layers]
    # create model
    model = tf.keras.Model(inputs=EMO_VGGFace2_model.inputs, outputs=output_layers)
    del EMO_VGGFace2_model
    tf.keras.backend.clear_session()
    return model

def get_EmoVGGFace2_embeddings_extractor(path_to_weights: str = None) -> tf.keras.Model:
    # construct old model
    EMO_VGGFace2_model = get_EMO_VGGFace2(path_to_weights)
    # start to construct new model by cutting off old one
    x = EMO_VGGFace2_model.get_layer('embeddings_layer').output
    # create model
    model = tf.keras.Model(inputs=EMO_VGGFace2_model.inputs, outputs=x)
    del EMO_VGGFace2_model
    tf.keras.backend.clear_session()
    return model




if __name__ == "__main__":
    path_to_weights = r'C:\Users\Denis\Downloads\Telegram Desktop\weights_0_66_37_affectnet_cat.h5'
    #model=get_EMO_VGGFace2(path_to_weights)
    model = get_pretrained_modified_EMO_VGGFace2(dense_neurons_after_conv=(1024, 512),
                                       dropout= 0.3, regularization=tf.keras.regularizers.l1_l2(0.0001),
                                       output_neurons = 7, pooling_at_the_end= 'avg',
                                       path_to_weights=path_to_weights)
    model.summary()
