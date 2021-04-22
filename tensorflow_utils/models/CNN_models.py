from typing import Tuple, List, Optional

import tensorflow as tf

from tensorflow_utils.models.resnet_blocks import resnet50_backend


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


'''def get_VGGFace2_resnet_model(input_shape:Tuple[int, int, int],
                           dense_neurons_after_conv:List[int,...],
                           dropout:float=0.3,
                           output_neurons:int=7, pooling_at_the_end:Optional[str]=None,
                           pretrained:bool=True,
                           path_to_weights:Optional[str]=None)->tf.keras.Model:
    # inputs are of size 224 x 224 x 3
    inputs = tf.keras.layers.Input(shape=input_dim, name='base_input')
    x = resnet50_backend(inputs)

    # pooling
    if pooling_at_the_end=='average':
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling_at_the_end=='max':
        x = tf.keras.layers.GlobalMaxPooling2D(name='global_max_pool')(x)
    # AvgPooling
    x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu', name='dim_proj')(x)

    if mode == 'train':
        y = keras.layers.Dense(nb_classes, activation='softmax',
                               use_bias=False, trainable=True,
                               kernel_initializer='orthogonal',
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               name='classifier_low_dim')(x)
    else:
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    # Compile
    model = keras.models.Model(inputs=inputs, outputs=y)
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    else:
        opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model'''