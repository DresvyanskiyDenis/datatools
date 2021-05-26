#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
from typing import Optional

import tensorflow as tf
import numpy as np


class Non_local_block_multi_head(tf.keras.layers.Layer):

    def __init__(self, num_heads:int,  output_channels:int,
                 head_output_channels:Optional[int]=None,
                 downsize_factor:Optional[int]=None,
                 shortcut_connection:bool=True,
                 relative_position_encoding:bool=False,
                 **kwargs):
        self.num_heads=num_heads
        if head_output_channels is None:
            self.head_output_channels=output_channels//num_heads
        else:
            self.head_output_channels=head_output_channels
        if downsize_factor is None:
            self.downsize_factor=num_heads
        else:
            self.downsize_factor=downsize_factor
        self.heads=[]
        self.output_channels=output_channels
        self.shortcut_connection=shortcut_connection
        self.relative_positional_encoding=relative_position_encoding
        super(Non_local_block_multi_head, self).__init__(**kwargs)

    def build(self, input_shape):
        # construct heads
        self.heads=[_Self_attention_non_local_block_without_shortcut_connection(output_channels=self.head_output_channels,
                                                                                downsize_factor= self.downsize_factor,
                                                                                mode='spatial', name_prefix="head_%i"%_,
                                                                                relative_position_encoding=self.relative_positional_encoding)
                    for _ in range(self.num_heads)]
        # invoke build() function for every head to construct them
        [head.build(input_shape) for head in self.heads]
        # construct output convolution and invoke its build() function
        self.output_conv=tf.keras.layers.Conv2D(self.output_channels, kernel_initializer='he_normal',
                                                 kernel_size=1, padding='same',use_bias=False)
        with tf.name_scope(name="multi_head_attention_output_conv"):
            self.output_conv.build((input_shape[0],input_shape[1],input_shape[2],self.head_output_channels*len(self.heads)))

        # save all trainable parameters in special variable
        for head_idx in range(self.num_heads):
            self._trainable_weights = self._trainable_weights + self.heads[head_idx].trainable_weights
        self._trainable_weights = self._trainable_weights+self.output_conv.trainable_weights

        # construct shortcut connection if needed and output number of channels differs from input number
        if self.shortcut_connection:
            if list(input_shape)[-1]!=self.output_channels:
                self.shortcut_conv=tf.keras.layers.Conv2D(self.output_channels, kernel_initializer='he_normal',
                                                 kernel_size=1, padding='same',use_bias=False)
                with tf.name_scope(name="multi_head_attention_shortcut_connection_conv"):
                    self.shortcut_conv.build(
                        (input_shape))
                self._trainable_weights=self._trainable_weights+self.shortcut_conv.trainable_weights

        # invoke the super.build() function as defined by keras authors
        super(Non_local_block_multi_head, self).build(input_shape)
        self.built=True

    def call(self, input):
        # go through all heads
        head_outputs=[self.heads[head_idx](input) for head_idx in range(self.num_heads)]
        # concatenate them
        concat_layer=tf.keras.layers.concatenate(head_outputs, axis=-1)
        # apply conv layer to concatenated outputs to get needed output size of filters at the end of layer
        output=self.output_conv(concat_layer)
        # apply shortcut connection if needed
        if self.shortcut_connection:
            if input.shape[-1]!=self.output_channels:
                shortcut=self.shortcut_conv(input)
            else:
                shortcut=input
            output=tf.keras.layers.Add()([shortcut, output])
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_channels
        return tuple(output_shape)


class _Self_attention_non_local_block_without_shortcut_connection(tf.keras.layers.Layer):
    """https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf

    """

    def __init__(self, output_channels:int, downsize_factor: int = 1, mode:str='spatio-temporal',
                 name_prefix:str="attention",relative_position_encoding:bool=False, **kwargs):
        """Initializes layer. Downsize_factor is needed to reduce the output number of channels
           by defined factor (by integer division on it)

        :param downsize_factor: int
                    see __init__ description
        :param kwargs:
        """
        if mode not in ('spatial', 'spatio-temporal'):
            raise AttributeError('Mode can be either \'spatial\' or \'spatio-temporal\'. Got %s.'%mode)
        self.mode=mode
        self.relative_pos_enc=relative_position_encoding
        # relative positional encoding is workable only with 'spatial' mode
        if self.relative_pos_enc:
            if self.mode!='spatial':
                raise AttributeError('Relative positional encoding works only with \'spatial\' mode.')
        self.name_prefix=name_prefix
        self.downsize_factor = downsize_factor
        self.output_channels=output_channels
        super(_Self_attention_non_local_block_without_shortcut_connection, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is either BxTxHxWxC or BxHxWxC
        # where B- batch_size, T - number of consecutive frames, H - height of image (feature map), W - width, C - number of channels
        # create convolutions for query, key, value and output
        if self.mode=="spatial":
            conv_layer=tf.keras.layers.Conv2D
            batch_size, height, width, channels = input_shape
        else:
            conv_layer=tf.keras.layers.Conv3D
            batch_size, num_images, height, width, channels = input_shape


        self.key_conv = conv_layer(input_shape[-1] // self.downsize_factor, kernel_initializer='he_normal',
                                               kernel_size=1, padding='same',use_bias=False)
        with tf.name_scope(name=self.name_prefix+"_key_conv"):
            self.key_conv.build(input_shape)


        self.query_conv = conv_layer(input_shape[-1] // self.downsize_factor, kernel_initializer='he_normal',
                                                 kernel_size=1, padding='same',use_bias=False)
        with tf.name_scope(name=self.name_prefix + "_query_conv"):
            self.query_conv.build(input_shape)


        self.value_conv = conv_layer(input_shape[-1] // self.downsize_factor, kernel_initializer='he_normal',
                                                 kernel_size=1, padding='same',use_bias=False)
        with tf.name_scope(name=self.name_prefix + "_value_conv"):
            self.value_conv.build(input_shape)


        self.output_conv=conv_layer(self.output_channels, kernel_initializer='he_normal',
                                                kernel_size=1, padding='same',use_bias=False)
        output_conv_input_shape=list(input_shape)
        output_conv_input_shape[-1]=output_conv_input_shape[-1]//self.downsize_factor
        with tf.name_scope(name=self.name_prefix + "_output_conv"):
            self.output_conv.build(tuple(output_conv_input_shape))

        # create batchNormalization layer
        self.bn_layer=tf.keras.layers.BatchNormalization()
        output_conv_output_shape= list(input_shape)
        output_conv_output_shape[-1] = self.output_channels
        with tf.name_scope(name=self.name_prefix + "_bn_after_output_conv"):
            self.bn_layer.build(output_conv_output_shape)

        # set trainable weights of this layer to be weights of all inner convolutional layers
        self._trainable_weights = self._trainable_weights+ self.key_conv.trainable_weights + self.query_conv.trainable_weights + \
                                  self.value_conv.trainable_weights + self.output_conv.trainable_weights + self.bn_layer.trainable_weights
        # add positional encoding if needed
        if self.relative_pos_enc:
            self.key_relative_w = self.add_weight(name=self.name_prefix+'_key_rel_w',
                                                  shape=[2 * width - 1, channels//self.downsize_factor],
                                                  initializer=tf.keras.initializers.RandomNormal(
                                                      stddev=channels ** -0.5))

            self.key_relative_h = self.add_weight(name=self.name_prefix+'_key_rel_h',
                                                  shape=[2 * height - 1, channels//self.downsize_factor],
                                                  initializer=tf.keras.initializers.RandomNormal(
                                                      stddev=channels ** -0.5))

        else:
            self.key_relative_w = None
            self.key_relative_h = None
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
        output_key_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]), name=self.name_prefix+"_reshape_key_conv")(x)
        # query flow
        x = self.query_conv(input)
        output_query_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]), name=self.name_prefix+"_reshape_query_conv")(x)
        # value flow
        x = self.value_conv(input)
        output_value_conv = tf.keras.layers.Reshape((-1, tf.keras.backend.int_shape(x)[-1]), name=self.name_prefix+"_reshape_value_conv")(x)
        # matrix multiplication for query and key
        multiplicated_key_query = tf.keras.layers.Dot(axes=2,name=self.name_prefix+"dot_key_query")([output_key_conv, output_query_conv])
        # relative position encoding
        if self.relative_pos_enc:
            shape_for_reshape = [1, height, width, output_query_conv.shape[-1]]
            reshaped_query = tf.keras.layers.Reshape(shape_for_reshape)(output_query_conv)
            h_rel_logits, w_rel_logits = self.relative_logits(reshaped_query)
            h_rel_logits = tf.keras.layers.Reshape(multiplicated_key_query.shape[1:])(h_rel_logits)
            w_rel_logits = tf.keras.layers.Reshape(multiplicated_key_query.shape[1:])(w_rel_logits)
            multiplicated_key_query = tf.keras.layers.Add()([multiplicated_key_query, h_rel_logits])
            multiplicated_key_query = tf.keras.layers.Add()([multiplicated_key_query, w_rel_logits])
        # softmax for obtained matrix
        softmax_output = tf.keras.layers.Softmax(name=self.name_prefix+"softmax_key_query")(multiplicated_key_query)
        # multiply value by obtained softmax matrix
        output_value = tf.keras.layers.Dot(axes=(2, 1),name=self.name_prefix+"dot_key_query_value")([softmax_output, output_value_conv])
        # reshape
        output = tf.keras.layers.Reshape(output_shape, name=self.name_prefix+"reshape_key_query_value")(output_value)
        # apply output conv
        output = self.output_conv(output)
        # batch normalization
        output=self.bn_layer(output)
        return output

    def relative_logits(self, q):
        shape = tf.keras.backend.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            tf.keras.backend.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = tf.keras.backend.reshape(rel_logits, [-1, 1 * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tf.keras.backend.reshape(rel_logits, [-1, 1, H, W, W])
        rel_logits = tf.keras.backend.expand_dims(rel_logits, axis=3)
        rel_logits = tf.keras.backend.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = tf.keras.backend.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = tf.keras.backend.reshape(rel_logits, [-1, 1, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = tf.keras.backend.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = tf.zeros(tf.keras.backend.stack([B, Nh, L, 1]))
        x = tf.keras.backend.concatenate([x, col_pad], axis=3)
        flat_x = tf.keras.backend.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = tf.zeros(tf.keras.backend.stack([B, Nh, L - 1]))
        flat_x_padded = tf.keras.backend.concatenate([flat_x, flat_pad], axis=2)
        final_x = tf.keras.backend.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_channels
        return tuple(output_shape)


class Self_attention_non_local_block(tf.keras.layers.Layer):
    """https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf

    """

    def __init__(self, output_channels:int, downsize_factor: int = 1,
                 mode:str='spatio-temporal', name_prefix:str="attention",
                 relative_position_encoding:bool=False, **kwargs):
        """Initializes layer. Downsize_factor is needed to reduce the output number of channels
           by defined factor (by integer division on it)

        :param downsize_factor: int
                    see __init__ description
        :param kwargs:
        """
        if mode not in ('spatial', 'spatio-temporal'):
            raise AttributeError('Mode can be either \'spatial\' or \'spatio-temporal\'. Got %s.'%mode)
        self.mode=mode
        self.relative_positional_encoding=relative_position_encoding
        self.downsize_factor = downsize_factor
        self.output_channels=output_channels
        self.name_prefix=name_prefix
        super(Self_attention_non_local_block, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is BxTxHxWxC,
        # where B- batch_size, T - number of consecutive frames, H - height of image (feature map), W - width, C - number of channels
        # create instance of _Self_attention_non_local_block_without_shortcut_connection, whcih has all needed operations
        # except shortcut connection
        self.non_local_block_without_shortcut=_Self_attention_non_local_block_without_shortcut_connection(self.output_channels,
                                                                                                          self.downsize_factor,
                                                                                                          self.mode,
                                                                                                          self.name_prefix,
                                                                                                          relative_position_encoding=self.relative_positional_encoding)
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
            with tf.name_scope(name=self.name_prefix + "shortcut_conv"):
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
        if self.output_channels != list(input.shape)[-1]:
            shortcut_connection=self.shortcut_conv(input)
        else:
            shortcut_connection=input
        # add shortcut connection to value
        output=tf.keras.layers.Add(name=self.name_prefix+"shortcut_add")([output_value, shortcut_connection])
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_channels
        return tuple(output_shape)





if __name__=="__main__":
    input_shape=(7,7,256)
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    model.add(Non_local_block_multi_head(num_heads=4,  output_channels=128,
                 head_output_channels=None,
                 downsize_factor=2, relative_position_encoding=True))
    model.add(Non_local_block_multi_head(num_heads=4,  output_channels=128,
                 head_output_channels=None,
                 downsize_factor=2, relative_position_encoding=True))
    '''model.add(Self_attention_non_local_block(output_channels=128,
                                             downsize_factor=2, mode='spatial', relative_position_encoding=True))'''
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape((-1,1)))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Flatten())
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True)
    x=np.random.uniform(-1,1, size=(100,7,7,256))
    y=np.ones((100,1))
    y[:5]=0
    permutations=np.random.permutation(x.shape[0])
    x, y = x[permutations], y[permutations]
    for i, w in enumerate(model.weights): print(i, w.name)
    model.fit(x,y, batch_size=10, epochs=5)
    model.save_weights('weights.h5')
    a=model.predict(x[:10])
    print('\n\n\n')


    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Input(input_shape))
    model1.add(Non_local_block_multi_head(num_heads=4, output_channels=128,
                                         head_output_channels=None,
                                         downsize_factor=2, relative_position_encoding=True))
    model1.add(Non_local_block_multi_head(num_heads=4, output_channels=128,
                                         head_output_channels=None,
                                         downsize_factor=2, relative_position_encoding=True))
    """model.add(Self_attention_non_local_block(output_channels=128,
                                             downsize_factor=2, mode='spatial'))"""
    model1.add(tf.keras.layers.Flatten())
    model1.add(tf.keras.layers.Reshape((-1, 1)))
    model1.add(tf.keras.layers.GlobalAveragePooling1D())
    model1.add(tf.keras.layers.Flatten())
    model1.load_weights('weights.h5')
    model1.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    b=model1.predict(x[:10])

    print(np.concatenate([a,b], axis=1))


