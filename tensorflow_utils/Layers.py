#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""

import tensorflow as tf
import numpy as np


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


class _Self_attention_non_local_block_without_shortcut_connection(tf.keras.layers.Layer):
    """https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf

    """

    def __init__(self, output_channels:int, downsize_factor: int = 1, mode:str='spatio-temporal', **kwargs):
        """Initializes layer. Downsize_factor is needed to reduce the output number of channels
           by defined factor (by integer division on it)

        :param downsize_factor: int
                    see __init__ description
        :param kwargs:
        """
        if mode not in ('spatial', 'spatio-temporal'):
            raise AttributeError('Mode can be either \'spatial\' or \'spatio-temporal\'. Got %s.'%mode)
        self.mode=mode
        self.downsize_factor = downsize_factor
        self.output_channels=output_channels
        super(_Self_attention_non_local_block_without_shortcut_connection, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is either BxTxHxWxC or BxHxWxC
        # where B- batch_size, T - number of consecutive frames, H - height of image (feature map), W - width, C - number of channels
        # create convolutions for query, key, value and output
        if self.mode=="spatial":
            conv_layer=tf.keras.layers.Conv2D
        else:
            conv_layer=tf.keras.layers.Conv3D

        self.key_conv = conv_layer(input_shape[-1] // self.downsize_factor, kernel_initializer='he_normal',
                                               kernel_size=1, padding='same',use_bias=False)
        self.key_conv.build(input_shape)


        self.query_conv = conv_layer(input_shape[-1] // self.downsize_factor, kernel_initializer='he_normal',
                                                 kernel_size=1, padding='same',use_bias=False)
        self.query_conv.build(input_shape)


        self.value_conv = conv_layer(input_shape[-1] // self.downsize_factor, kernel_initializer='he_normal',
                                                 kernel_size=1, padding='same',use_bias=False)
        self.value_conv.build(input_shape)


        self.output_conv=conv_layer(self.output_channels, kernel_initializer='he_normal',
                                                kernel_size=1, padding='same',use_bias=False)
        output_conv_input_shape=list(input_shape)
        output_conv_input_shape[-1]=output_conv_input_shape[-1]//self.downsize_factor
        self.output_conv.build(tuple(output_conv_input_shape))

        # create batchNormalization layer
        self.bn_layer=tf.keras.layers.BatchNormalization()
        output_conv_output_shape= list(input_shape)
        output_conv_output_shape[-1] = self.output_channels
        self.bn_layer.build(output_conv_output_shape)

        # set trainable weights of this layer to be weights of all inner convolutional layers
        self._trainable_weights = self.key_conv.trainable_weights + self.query_conv.trainable_weights + \
                                  self.value_conv.trainable_weights + self.output_conv.trainable_weights + self.bn_layer.trainable_weights
        # invoke the super.build() function as defined by keras authors
        super(_Self_attention_non_local_block_without_shortcut_connection, self).build(input_shape)
        self.built = True

    def call(self, input):
        # extract shapes
        if self.mode=='spatial':
            batch_size, height, width, channels = input.shape
            output_shape=(height, width, channels//self.downsize_factor)
        else:
            batch_size,temporal_axis, height, width, channels = input.shape
            output_shape = (temporal_axis,height, width, channels//self.downsize_factor)
        # key flow
        x = self.key_conv(input)
        output_key_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]))(x)
        # query flow
        x = self.query_conv(input)
        output_query_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]))(x)
        # value flow
        x = self.value_conv(input)
        output_value_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]))(x)
        # matrix multiplication for query and key
        multiplicated_key_query = tf.keras.layers.Dot(axes=2)([output_key_conv, output_query_conv])
        # softmax for obtained matrix
        softmax_output = tf.keras.layers.Softmax()(multiplicated_key_query)
        # multiply value by obtained softmax matrix
        output_value = tf.keras.layers.Dot(axes=(2, 1))([softmax_output, output_value_conv])
        # reshape
        output = tf.keras.layers.Reshape(output_shape)(output_value)
        # apply output conv
        output = self.output_conv(output)
        # batch normalization
        output=self.bn_layer(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_channels
        return tuple(output_shape)


class Self_attention_non_local_block(tf.keras.layers.Layer):
    """https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf

    """

    def __init__(self, output_channels:int, downsize_factor: int = 1, mode:str='spatio-temporal',**kwargs):
        """Initializes layer. Downsize_factor is needed to reduce the output number of channels
           by defined factor (by integer division on it)

        :param downsize_factor: int
                    see __init__ description
        :param kwargs:
        """
        if mode not in ('spatial', 'spatio-temporal'):
            raise AttributeError('Mode can be either \'spatial\' or \'spatio-temporal\'. Got %s.'%mode)
        self.mode=mode
        self.downsize_factor = downsize_factor
        self.output_channels=output_channels
        super(Self_attention_non_local_block, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is BxTxHxWxC,
        # where B- batch_size, T - number of consecutive frames, H - height of image (feature map), W - width, C - number of channels
        # create instance of _Self_attention_non_local_block_without_shortcut_connection, whcih has all needed operations
        # except shortcut connection
        self.non_local_block_without_shortcut=_Self_attention_non_local_block_without_shortcut_connection(self.output_channels,
                                                                                                          self.downsize_factor,
                                                                                                          self.mode)
        self.non_local_block_without_shortcut.build(input_shape)
        self._trainable_weights =self.non_local_block_without_shortcut.trainable_weights
        # create 1x1x1 conv in case the amount of output channels will differ from amount of input channels (to make it similar and permord add operation)
        if self.output_channels!=list(input_shape)[-1]:
            if self.mode=='spatial':
                self.shortcut_conv=tf.keras.layers.Conv2D(self.output_channels, kernel_initializer='he_normal',
                                                kernel_size=1, padding='same', use_bias=False)
            else:
                self.shortcut_conv = tf.keras.layers.Conv3D(self.output_channels, kernel_initializer='he_normal',
                                                kernel_size=1, padding='same', use_bias=False)
            self.shortcut_conv.build(input_shape)
            self._trainable_weights=self._trainable_weights+self.shortcut_conv.trainable_weights
        # set trainable weights of this layer to be weights of all inner convolutional layers
        # invoke the super.build() function as defined by keras authors
        super(Self_attention_non_local_block, self).build(input_shape)
        self.built = True

    def call(self, input):
        # attention mechanism
        output_value=self.non_local_block_without_shortcut(input)
        # shortcut connection
        if self.output_channels != list(input_shape)[-1]:
            shortcut_connection=self.shortcut_conv(input)
        else:
            shortcut_connection=input
        # add shortcut connection to value
        output=tf.keras.layers.Add()([output_value, shortcut_connection])
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_channels
        return tuple(output_shape)





if __name__=="__main__":
    input_shape=(10,50,50,128)
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    model.add(Self_attention_non_local_block(downsize_factor=2, output_channels=64))
    model.add(Self_attention_non_local_block( downsize_factor=2, output_channels=32))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape((-1,1)))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Flatten())
    model.compile(optimizer='Adam', loss='mse')
    model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True)
    x=np.random.uniform(-1,1, size=(10,10, 50,50,128))
    y=np.ones((10,1))
    y[:5]=0
    permutations=np.random.permutation(x.shape[0])
    x, y = x[permutations], y[permutations]
    model.fit(x,y, batch_size=1, epochs=100)


