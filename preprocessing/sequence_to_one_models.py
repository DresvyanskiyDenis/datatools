#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains various building scripts for sequence-to-one deep models
For model constructions the tensorflow library is used.

tensorflow: https://www.tensorflow.org

List of functions:
    * create_1d_cnn_model_classification - builds 1D-CNN model for sequence-to-one classification
    * create_simple_RNN_network - builds RNN-network (LSTM, GRU or simple rnn) for sequence-to-one classification
    * chunk_based_rnn_model - builds chunk-based RNN model. Chunk-based means that the input shape of the model is
    (num_chunks, num_windows, num_features) - 3D instead of 2D usually used fot RNN modelling.
    * chunk_based_rnn_attention_model - builds chunk-based RNN model with Attention layer at the end of network.
    Chunk-based means that the input shape of the model is (num_chunks, num_windows, num_features) - 3D instead
    of 2D usually used fot RNN modelling.
    * chunk_based_1d_cnn_attention_model - builds chunk-based 1D CNN model with Attention layer at the end of network.
    Chunk-based means that the input shape of the model is (num_chunks, num_windows, num_features) - 3D instead
    of 2D usually used fot RNN modelling.
    * ccc_loss - Concordance Correlation Coefficient loss taken from AVEC scripts.
    * CCC_loss_tf - Concordance Correlation Coefficient loss taken, my own tensorflow implementation.
"""
from typing import Tuple, Union, Optional
import tensorflow as tf
import tensorflow.keras.backend as K

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from attention import Attention


def create_1d_cnn_model_classification(*, input_shape: Tuple[int, ...], num_classes: int,
                                       kernel_sizes: Tuple[int, ...] = (15, 15, 12, 12, 10, 10, 5, 5, 4, 3),
                                       filter_numbers: Tuple[int, ...] = (16, 32, 64, 64, 128, 128, 256, 256, 512, 512),
                                       pooling_step: Optional[int] = 2,
                                       need_regularization: bool = False) -> tf.keras.Model:
    """ Creates 1D CNN model according to provided parameters

    :param input_shape:tuple
                    input shape for tensrflow.keras model
    :param num_classes: int
    :param kernel_sizes: list
                    list of kernel sizes
    :param filter_numbers:list
                    list of number of filters, usually in descending order
    :param pooling_step: int
                    the step after which the pooling operation will be used
                    e. g. poling_step=2 means that every 2 layers the pooling operation will be applied
    :param need_regularization: bool
                    use regularization in conv layers or not
                    if true, tf.keras.regularizers.l2(1e-5) will be applied
    :return:
    """
    # if length of kernel_sizes and filter_numbers are not equal, raise exception
    if len(kernel_sizes) != len(filter_numbers):
        raise AttributeError(
            'lengths of kernel_sizes and filter_numbers must be equal! Got kernel_sizes:%i, filter_numbers:%i.' % (
            len(kernel_sizes), len(filter_numbers)))
    # create input layer for model
    input = tf.keras.layers.Input(input_shape)
    x = input

    for layer_idx in range(len(kernel_sizes)):
        # define parameters for convenience visibility
        filters = filter_numbers[layer_idx]
        kernel_size = kernel_sizes[layer_idx]
        regularization = tf.keras.regularizers.l2(1e-5) if need_regularization else None
        # create Conv1D layer
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
                                   kernel_regularizer=regularization)(x)
        # if now is a step of pooling layer, then create and add to model AveragePooling layer
        if (layer_idx + 1) % pooling_step == 0:
            x = tf.keras.layers.AveragePooling1D(2)(x)
    # apply GlobalAveragePoling to flatten output of conv1D
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # apply Dense layer and then output softmax layer with num_class numbers neurons
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    # create a model and return it as a result of function
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def create_simple_RNN_network(*, input_shape: Tuple[int, ...], num_output_neurons: int,
                              neurons_on_layer: Tuple[int, ...] = (256, 256),
                              rnn_type: str = 'LSTM',
                              dnn_layers: Tuple[int, ...],
                              need_regularization: bool = False,
                              dropout: bool = False) -> tf.keras.Model:
    """ Creates RNN neural network according to provided parameters

    :param input_shape: tuple
                    input shape (of sequence )for tensrflow.keras model
    :param num_output_neurons: int
                    amount of neurons at the end of model (output of model)
    :param neurons_on_layer: tuple(int,...)
                    list of numbers of neurons on each layer. It controls the depth of the network as well.
    :param rnn_type: str
                    can be either 'simple', 'LSTM' or 'GRU'. Specifies the type of reccurent layer, which will be used
    :param dnn_layers: tuple(str,...)
                    list of numbers of neurons on each layer right after reccurent layers. Can be empty ().
    :param need_regularization: bool
                    Specifies if it is needed to use regularization in reccurent and dense layers or not
                    if true, tf.keras.regularizers.l2(1e-5) will be applied
    :param dropout: bool
                    Specifies, if it is needed to use dropour after every reccurent and dense layers.
                    if true, tf.keras.layers.Dropout(0.2) will be applied
    :return:
    """
    # create input layer for Model
    if len(input_shape) != 2:
        raise AttributeError('input shape must be 2-dimensional. Got %i' % (len(input_shape)))
    # define rnn rnn_type
    if rnn_type == 'simple':
        layer_type = tf.keras.layers.SimpleRNN
    elif rnn_type == 'LSTM':
        layer_type = tf.keras.layers.LSTM
    elif rnn_type == 'GRU':
        layer_type = tf.keras.layers.GRU
    else:
        raise AttributeError('rnn_type should be either \'simple\', \'LSTM\' or \'GRU\'. Got %s' % (rnn_type))
    regularization = tf.keras.regularizers.l2(1e-5) if need_regularization else None

    input = tf.keras.layers.Input(input_shape)
    x = input
    for layer_idx in range(len(neurons_on_layer) - 1):
        neurons = neurons_on_layer[layer_idx]
        x = layer_type(neurons, return_sequences=True, kernel_regularizer=regularization)(x)
        if dropout: x = tf.keras.layers.Dropout(0.2)(x)
    # last RNN layer
    x = layer_type(neurons_on_layer[-1])(x)
    # dnn layers
    for layer_idx in range(len(dnn_layers)):
        neurons = dnn_layers[layer_idx]
        x = tf.keras.layers.Dense(neurons, activation='relu', kernel_regularizer=regularization)(x)
        if dropout: x = tf.keras.layers.Dropout(0.2)(x)
    # last layer
    output = tf.keras.layers.Dense(num_output_neurons, activation='tanh')(x)
    output = tf.keras.layers.Reshape((-1, 1))(output)
    # create model
    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model


def chunk_based_rnn_model(*, input_shape: Tuple[int, ...], num_output_neurons: int,
                          neurons_on_layer: Tuple[int, ...] = (256, 256),
                          rnn_type: str = 'LSTM',
                          bidirectional: bool = False,
                          need_regularization: bool = False,
                          dropout: bool = False,
                          dropout_rate=0.3
                          ) -> tf.keras.Model:
    """Builds chunk based rnn model, which schematically can be showed as
            TimeDistributed(RNN) - Conv2D (1x1) - RNN - Dense

    :param input_shape: Tuple(int,...)
                the input shape of the model
    :param num_output_neurons: int
                the number of neurons on the last layer. Usually, it equals to the number of classes.
    :param neurons_on_layer: tuple(int,...)
                list of numbers of neurons on each layer. It controls the depth of the network as well.
    :param rnn_type:str
                can be either 'simple', 'LSTM' or 'GRU'. Specifies the type of reccurent layer, which will be used
    :param bidirectional: bool
                should the rnn layers be bidirectional or not
    :param need_regularization: bool
                Specifies if it is needed to use regularization in reccurent and dense layers or not
                if true, tf.keras.regularizers.l2(1e-5) will be applied
    :param dropout: bool
                if True, locate dropout layers after rnn and 2D convolutional layers
    :param dropout_rate: float
                the rte of dropout layer
    :return: tf.keras.Model
                built, but not compiled model

    """
    if len(input_shape) != 3:
        raise AttributeError('input shape must be 3-dimensional. Got %i' % (len(input_shape)))
    # define rnn rnn_type
    if rnn_type == 'simple':
        layer_type = tf.keras.layers.SimpleRNN
    elif rnn_type == 'LSTM':
        layer_type = tf.keras.layers.LSTM
    elif rnn_type == 'GRU':
        layer_type = tf.keras.layers.GRU

    else:
        raise AttributeError('rnn_type should be either \'simple\', \'LSTM\' or \'GRU\'. Got %s' % (rnn_type))
    regularization = tf.keras.regularizers.l2(1e-5) if need_regularization else None

    # building neural network
    input = tf.keras.layers.Input(input_shape)
    # rnn part
    x = input
    for layer_idx in range(len(neurons_on_layer)):
        neurons = neurons_on_layer[layer_idx]
        if bidirectional:
            reccurent_layer = tf.keras.layers.Bidirectional(
                layer_type(neurons, return_sequences=True, kernel_regularizer=regularization))
        else:
            reccurent_layer = layer_type(neurons, return_sequences=True, kernel_regularizer=regularization)
        x = tf.keras.layers.TimeDistributed(reccurent_layer)(x)
        if dropout: x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate))(x)
    #  average the last hidden states from different chunks with the help of 1x1 Conv2d
    x = tf.keras.layers.Conv2D(128, 1, activation='tanh', kernel_regularizer=regularization)(x)
    if dropout: x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='tanh')(x)
    # squeeze the last dimension
    x = tf.keras.layers.Reshape((x.shape[1:-1]))(x)
    # couple rnn layers on sentence level
    x = layer_type(128, return_sequences=True, kernel_regularizer=regularization)(x)
    if dropout: x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = layer_type(128, kernel_regularizer=regularization)(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(num_output_neurons, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model


def chunk_based_rnn_attention_model(*, input_shape: Tuple[int, ...], num_output_neurons: int,
                                    neurons_on_rnn_layer: Tuple[int, ...] = (256, 256),
                                    rnn_type: str = 'LSTM',
                                    bidirectional: bool = False,
                                    need_regularization: bool = False,
                                    dropout: bool = False,
                                    dropout_rate=0.3
                                    ) -> tf.keras.Model:
    """Builds chunk based rnn model with attention layer, which schematically can be showed as
                TimeDistributed(RNN) - Conv2D (1x1) - RNN - Attention - Dense

        :param input_shape: Tuple(int,...)
                    the input shape of the model
        :param num_output_neurons: int
                    the number of neurons on the last layer. Usually, it equals to the number of classes.
        :param neurons_on_layer: tuple(int,...)
                    list of numbers of neurons on each layer. It controls the depth of the network as well.
        :param rnn_type:str
                    can be either 'simple', 'LSTM' or 'GRU'. Specifies the type of reccurent layer, which will be used
        :param bidirectional: bool
                    should the rnn layers be bidirectional or not
        :param need_regularization: bool
                    Specifies if it is needed to use regularization in reccurent and dense layers or not
                    if true, tf.keras.regularizers.l2(1e-5) will be applied
        :param dropout: bool
                    if True, locate dropout layers after rnn and 2D convolutional layers
        :param dropout_rate: float
                    the rte of dropout layer
        :return: tf.keras.Model
                    built, but not compiled model

        """
    if len(input_shape) != 3:
        raise AttributeError('input shape must be 3-dimensional. Got %i' % (len(input_shape)))
    # define rnn rnn_type
    if rnn_type == 'simple':
        layer_type = tf.keras.layers.SimpleRNN
    elif rnn_type == 'LSTM':
        layer_type = tf.keras.layers.LSTM
    elif rnn_type == 'GRU':
        layer_type = tf.keras.layers.GRU
    else:
        raise AttributeError('rnn_type should be either \'simple\', \'LSTM\' or \'GRU\'. Got %s' % (rnn_type))

    regularization = tf.keras.regularizers.l2(1e-5) if need_regularization else None
    # building neural network
    input = tf.keras.layers.Input(input_shape)
    # rnn part
    x = input
    for layer_idx in range(len(neurons_on_rnn_layer)-1):
        neurons = neurons_on_rnn_layer[layer_idx]
        if bidirectional:
            reccurent_layer = tf.keras.layers.Bidirectional(
                layer_type(neurons, return_sequences=True, kernel_regularizer=regularization))
        else:
            reccurent_layer = layer_type(neurons, return_sequences=True, kernel_regularizer=regularization)
        x = tf.keras.layers.TimeDistributed(reccurent_layer)(x)
        if dropout: x = tf.keras.layers.Dropout(dropout_rate)(x)
    # last rnn layer
    neurons = neurons_on_rnn_layer[-1]
    if bidirectional:
        reccurent_layer = tf.keras.layers.Bidirectional(
            layer_type(neurons, return_sequences=False, kernel_regularizer=regularization))
    else:
        reccurent_layer = layer_type(neurons, return_sequences=False, kernel_regularizer=regularization)
    x = tf.keras.layers.TimeDistributed(reccurent_layer)(x)
    if dropout: x = tf.keras.layers.Dropout(dropout_rate)(x)
    # squeeze the last dimension
    #x = tf.keras.layers.Reshape((x.shape[1:-1]))(x)
    # average the second dimension - we will get averaged by timesteps results for every 'features' (last channel)
    #x = layer_type(128, return_sequences=True, kernel_regularizer=regularization)(x)
    # attention part
    x = Attention(128)(x)
    # output
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(num_output_neurons, activation='softmax')(x)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model


def chunk_based_1d_cnn_attention_model(*, input_shape: Tuple[int, ...], num_output_neurons: int,
                                       filters_per_layer: Tuple[int, ...] = (64, 128, 256, 512),
                                       filter_sizes: Tuple[int, ...] = (10, 8, 6, 5),
                                       pooling_sizes: Tuple[int, ...] = (2, 2),
                                       pooling_step: Optional[int] = 2,
                                       need_regularization: bool = False,
                                       dropout: bool = False,
                                       dropout_rate=0.3
                                       ) -> tf.keras.Model:
    """Builds chunk based 1d cnn model with attention layer at the end. The model van be schematically showed as
            TimeDistributed(1D CNN) - Conv2D (1x1) - Attention - Dense

    :param input_shape: Tuple[int,...]
                the input shape of the model
    :param num_output_neurons: int
                the number of neurons on the last layer. Usually, it equals to the number of classes.
    :param filters_per_layer: Tuple[int,...]
                the number of the filter on each layer
    :param filter_sizes: Tuple[int,...]
                the sizes of the filters on each layer. Shape shoul be equal to the fitlers_per_layer
    :param pooling_sizes: Tuple[int,...]
                the sizes of the pooling
    :param pooling_step: int
                the number of layers, after which pooling must be applied
    :param need_regularization: bool
                Specifies if it is needed to use regularization in reccurent and dense layers or not
                if true, tf.keras.regularizers.l2(1e-5) will be applied
    :param dropout: bool
                if True, locate dropout layers after rnn and 2D convolutional layers
    :param dropout_rate: float
                the rte of dropout layer
    :return: tf.keras.Model
                built, but not compiled model
    """
    # create input layer for model
    input = tf.keras.layers.Input(input_shape)
    x = input
    pooling_idx = 0
    for layer_idx in range(len(filters_per_layer)):
        # define parameters for convenience visibility
        filters = filters_per_layer[layer_idx]
        kernel_size = filter_sizes[layer_idx]
        regularization = tf.keras.regularizers.l2(1e-5) if need_regularization else None
        # create Conv1D layer
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
                                   kernel_regularizer=regularization))(x)
        # add dropout
        if dropout: x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate))(x)
        # if now is a step of pooling layer, then create and add to model AveragePooling layer
        if (layer_idx + 1) % pooling_step == 0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling1D(pooling_sizes[pooling_idx]))(x)
            pooling_idx += 1
    #  average pooling to squeeze tensor to 2-d
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D())(x)
    # attention
    x = Attention(128)(x)
    #x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    #if dropout: x=tf.keras.layers.Dropout(dropout_rate)(x)
    #x = tf.keras.layers.LSTM(128, return_sequences=False)(x)
    #if dropout: x = tf.keras.layers.Dropout(dropout_rate)(x)
    # output
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_output_neurons, activation='softmax')(x)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model


def ccc_loss(gold,
             pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    gold = K.squeeze(gold, axis=-1)
    pred = K.squeeze(pred, axis=-1)
    gold_mean = K.mean(gold, axis=-1, keepdims=True)
    pred_mean = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold - gold_mean) * (pred - pred_mean)
    gold_var = K.mean(K.square(gold - gold_mean), axis=-1, keepdims=True)
    pred_var = K.mean(K.square(pred - pred_mean), axis=-1, keepdims=True)
    ccc = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.epsilon())
    ccc_loss = K.constant(1.) - ccc
    return ccc_loss


def CCC_loss_tf(y_true, y_pred):
    """
    This function calculates loss based on concordance correlation coefficient of two series: 'ser1' and 'ser2'
    TensorFlow methods are used
    """
    # tf.print('y_true_shape:',tf.shape(y_true))
    # tf.print('y_pred_shape:',tf.shape(y_pred))

    y_true_mean = K.mean(y_true, axis=-2, keepdims=True)
    y_pred_mean = K.mean(y_pred, axis=-2, keepdims=True)

    y_true_var = K.mean(K.square(y_true - y_true_mean), axis=-2, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred - y_pred_mean), axis=-2, keepdims=True)

    cov = K.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=-2, keepdims=True)

    ccc = tf.math.multiply(2., cov) / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    ccc_loss = 1. - K.mean(K.flatten(ccc))
    # tf.print('ccc:', tf.shape(ccc_loss))
    # tf.print('ccc_loss:',ccc_loss)
    return ccc_loss


if __name__ == "__main__":
    model=chunk_based_1d_cnn_attention_model(input_shape=(24,8000,1), num_output_neurons=3,
                                       filters_per_layer = (64, 128, 128, 256,256),
                                       filter_sizes = (12, 10, 8, 6, 5),
                                       pooling_sizes = (8, 4,2,2,2),
                                       pooling_step = 1,
                                       need_regularization= True,
                                       dropout= True,
                                       dropout_rate=0.3
                                       )
    model.summary()
