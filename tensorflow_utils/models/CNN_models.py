from typing import Tuple, List, Optional

import tensorflow as tf

def get_mobilenet_v2_model(input_shape:Tuple[int, int, int],
                           dense_neurons_after_conv:List[int,...],
                           dropout:float=0.3,
                           output_neurons:int=7, pooling_at_the_end:Optional[str]=None,
                           pretrained:bool=True)->tf.keras.Model:
    if pretrained:
        weights='imagenet'
    else:
        weights=None
    mobilenet_v2_model=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                                      include_top=False,
                                                                      weights=weights,
                                                                      pooling=pooling_at_the_end)
    x=mobilenet_v2_model.output
    for idx_dense_layer_neurons in range(len(dense_neurons_after_conv)-1):
        neurons=dense_neurons_after_conv[idx_dense_layer_neurons]
        x = tf.keras.layers.Dense(neurons, activation='relu')(x)
        if dropout: x = tf.keras.layers.Dropout(dropout)(x)
    # last layer
    x = tf.keras.layers.Dense(dense_neurons_after_conv[-1], activation='relu')(x)
    output=tf.keras.layers.Dense(output_neurons, activation='softmax')(x)
    result_model=tf.keras.Model(inputs=mobilenet_v2_model.inputs, outputs=[output])
    return result_model