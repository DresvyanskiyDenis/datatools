#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""

import tensorflow as tf
import numpy as np


class Self_attention_pixel_wise(tf.keras.layers.Layer):
    """Represents pixel-wise convolutional self-attention layer, which is also called by authors non-local block
    https://arxiv.org/pdf/1711.07971.pdf
    It uses inheritance from tf.keras.layers.Layer.
    """
    def __init__(self, downsize_factor:int=1, **kwargs):
        """Initializes layer. Downsize_factor is needed to reduce the output number of channels
           by defined factor (by integer division on it)

        :param downsize_factor: int
                    see __init__ description
        :param kwargs:
        """
        self.downsize_factor=downsize_factor
        super(Self_attention_pixel_wise, self).__init__(**kwargs)

    def build(self, input_shape):
        # create 1x1 convolutions for query, key, value and output
        self.key_conv=tf.keras.layers.Conv2D(input_shape[-1]//self.downsize_factor, kernel_size=1, padding='same')
        self.key_conv.build(input_shape)
        self.query_conv = tf.keras.layers.Conv2D(input_shape[-1] // self.downsize_factor, kernel_size=1, padding='same')
        self.query_conv.build(input_shape)
        self.value_conv = tf.keras.layers.Conv2D(input_shape[-1] // self.downsize_factor, kernel_size=1, padding='same')
        self.value_conv.build(input_shape)
        self.output_conv=tf.keras.layers.Conv2D(input_shape[-1], kernel_size=1, padding='same')
        self.output_conv.build((input_shape[0],input_shape[1],input_shape[2],input_shape[-1]//self.downsize_factor))
        # set trainable weights of this layer to be weights of all inner 1D convolutional layers
        self._trainable_weights=self.key_conv.trainable_weights+self.query_conv.trainable_weights+\
                                self.value_conv.trainable_weights+self.output_conv.trainable_weights
        # invoke the super.build() function as defined by keras authors
        super(Self_attention_pixel_wise, self).build(input_shape)
        self.built=True

    def call(self, input):
        # extract shapes
        batch_size, height, width, channels=input.shape
        # key flow
        x = self.key_conv(input)
        output_key_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]))(x)
        # query flow
        x = self.query_conv(input)
        output_query_conv = tf.keras.layers.Reshape(( -1, tf.keras.backend.int_shape(x)[-1]))(x)
        output_query_conv_transpose=tf.transpose(output_query_conv, perm=[0,2,1])
        # value flow
        x = self.value_conv(input)
        output_value_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]))(x)
        # matrix multiplication for query and key
        multiplicated_key_query=tf.keras.layers.Dot(axes=(2,1))([output_key_conv, output_query_conv_transpose])
        # softmax for obtained matrix
        softmax_output=tf.keras.layers.Softmax()(multiplicated_key_query)
        # multiply value by obtained softmax matrix
        output_value=tf.keras.layers.Dot(axes=(2,1))([softmax_output, output_value_conv])
        # reshape and apply output conv
        output_value = tf.keras.layers.Reshape(( height, width,
                                                tf.keras.backend.int_shape(output_value)[-1]))(output_value)
        output_value=self.output_conv(output_value)
        # add input to the output value (shortcut connection)
        output = tf.keras.layers.Add()([output_value, input])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2],input_shape[3]//self.downsize_factor)


#class Multi_head_self_attention_pixel_wise(tf.keras.layers.Layer):




if __name__=="__main__":
    input_shape=(50,50,128)
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    model.add(Self_attention_pixel_wise())
    model.add(Self_attention_pixel_wise())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape((-1,1)))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Flatten())
    model.compile(optimizer='Adam', loss='mse')
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
    x=np.random.uniform(-1,1, size=(120,50,50,128))
    y=np.ones((120,1))
    y[:60]=0
    permutations=np.random.permutation(x.shape[0])
    x, y = x[permutations], y[permutations]
    model.fit(x,y, batch_size=10, epochs=100)


