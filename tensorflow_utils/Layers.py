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
        # create self-attention layer (see this class below)
        self.self_attention_layer=_Self_attention_pixel_wise_without_shortcut(self.downsize_factor)
        self.self_attention_layer.build(input_shape)
        self._trainable_weights=self.self_attention_layer.trainable_weights
        # reduce shape of shortcut if needed
        if self.downsize_factor>1:
            self.reduce_shortcut_conv=tf.keras.layers.Conv2D(input_shape[-1]//self.downsize_factor, kernel_size=1, padding='same')
            self.reduce_shortcut_conv.build(input_shape)
            self._trainable_weights+=self.reduce_shortcut_conv.trainable_weights
        # invoke the super.build() function as defined by keras authors
        super(Self_attention_pixel_wise, self).build(input_shape)
        self.built=True

    def call(self, input):
        # extract shapes
        batch_size, height, width, channels=input.shape
        attention_output=self.self_attention_layer(input)
        # add input to the output value (shortcut connection)
        if self.downsize_factor>1:
            shortcut=self.reduce_shortcut_conv(input)
            output=tf.keras.layers.Add()([attention_output, shortcut])
        else:
            output = tf.keras.layers.Add()([attention_output, input])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2],input_shape[3]//self.downsize_factor)


class _Self_attention_pixel_wise_without_shortcut(tf.keras.layers.Layer):
    """The same attention layer as Self_attention_pixel_wise, but without shortcut connection.
    It is needed for constructing multi-head convolution self-attention (see below).
    """
    def __init__(self, downsize_factor:int=1, **kwargs):
        """Initializes layer. Downsize_factor is needed to reduce the output number of channels
           by defined factor (by integer division on it)

        :param downsize_factor: int
                    see __init__ description
        :param kwargs:
        """
        self.downsize_factor=downsize_factor
        super(_Self_attention_pixel_wise_without_shortcut, self).__init__(**kwargs)

    def build(self, input_shape):
        # create 1x1 convolutions for query, key, value and output
        self.key_conv=tf.keras.layers.Conv2D(input_shape[-1]//self.downsize_factor, kernel_size=1, padding='same')
        self.key_conv.build(input_shape)
        self.query_conv = tf.keras.layers.Conv2D(input_shape[-1] // self.downsize_factor, kernel_size=1, padding='same')
        self.query_conv.build(input_shape)
        self.value_conv = tf.keras.layers.Conv2D(input_shape[-1] // self.downsize_factor, kernel_size=1, padding='same')
        self.value_conv.build(input_shape)
        # set trainable weights of this layer to be weights of all inner 1D convolutional layers
        self._trainable_weights=self.key_conv.trainable_weights+self.query_conv.trainable_weights+\
                                self.value_conv.trainable_weights
        # invoke the super.build() function as defined by keras authors
        super(_Self_attention_pixel_wise_without_shortcut, self).build(input_shape)
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
        # reshape
        output = tf.keras.layers.Reshape((height, width,
                                        tf.keras.backend.int_shape(output_value)[-1]))(output_value)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2],input_shape[3]//self.downsize_factor)


class Multi_head_self_attention_pixel_wise(tf.keras.layers.Layer):

    def __init__(self, num_heads:int, output_filters:int, downsize_factor:int=8,  **kwargs):
        self.num_heads=num_heads
        self.downsize_factor=downsize_factor
        self.heads=[]
        self.output_filters=output_filters
        super(Multi_head_self_attention_pixel_wise, self).__init__(**kwargs)

    def build(self, input_shape):
        # construct heads
        self.heads=[_Self_attention_pixel_wise_without_shortcut(self.downsize_factor) for _ in range(self.num_heads)]
        # invoke build() function for every head to construct them
        [head.build(input_shape) for head in self.heads]
        # save all trainable parameters in special variable
        for head_idx in range(self.num_heads):
            self._trainable_weights=self._trainable_weights+self.heads[head_idx].trainable_weights
        # construct output convolution and invoke its build() function
        self.output_conv=tf.keras.layers.Conv1D(self.output_filters, 1, padding='same')
        self.output_conv.build((input_shape[0],input_shape[1],input_shape[2],input_shape[3]*len(self.heads)//self.downsize_factor))
        # invoke the super.build() function as defined by keras authors
        super(Multi_head_self_attention_pixel_wise, self).build(input_shape)
        self.built=True

    def call(self, input):
        # go through all heads
        head_outputs=[self.heads[head_idx](input) for head_idx in range(self.num_heads)]
        # concatenate them
        concat_layer=tf.keras.layers.concatenate(head_outputs, axis=-1)
        # apply 1x1 conv layer to concatenated outputs to get needed output size of filters at the end of layer
        output=self.output_conv(concat_layer)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2],self.output_shape)




if __name__=="__main__":
    input_shape=(50,50,128)
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    model.add(Self_attention_pixel_wise(downsize_factor=2))
    model.add(Self_attention_pixel_wise( downsize_factor=2))
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
    model.fit(x,y, batch_size=4, epochs=100)


