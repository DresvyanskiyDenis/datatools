from typing import Tuple, Optional, List, Callable

import pandas as pd
import numpy as np
import tensorflow as tf

Tensorflow_Callable = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def get_tensorflow_sequence_loader(embeddings_and_labels: pd.DataFrame, num_classes: int,
                                   batch_size: int, type_of_labels: str,
                                   window_size: int, window_shift:int, window_stride: int,
                                   shuffle: bool = True,
                             preprocessing_function: Optional[Tensorflow_Callable] = None,
                             clip_values: Optional[bool] = None,
                             cache_loaded_seq:Optional[bool]=None) -> tf.data.Dataset:
    """TODO: write the function description

    :param embeddings_and_labels: pd.DataFrame
            dataframe with columns ["embedding_0", "embedding_1", ..., "label_0", "label_1", ...]
    :param num_classes: int
    :param batch_size: int
    :param type_of_labels: str
    :param window_size: int
    :param window_shift: int
    :param window_stride: int
    :param shuffle: bool
    :param preprocessing_function: Optional[Tensorflow_Callable]
    :param clip_values: Optional[bool]
    :param cache_loaded_seq: Optional[bool]
    """
    AUTOTUNE = tf.data.AUTOTUNE
    # create tf.data.Dataset from provided paths to the images and labels
    dataset = tf.data.Dataset.from_tensor_slices(embeddings_and_labels)
    # cache for better performance if specified
    if cache_loaded_seq:
        dataset = dataset.cache()
    # forming sequences of windows
    dataset = dataset.window(window_size, shift=window_shift, stride=window_stride)
    # commant to convert windows from the Dataset instances to the Tensors back to unite them in one Dataset
    dataset = dataset.flat_map(lambda x:x.batch(window_size, drop_remainder=True))
    # divide features and labels
    def divide_features_and_labels(x):
        return (x[:, :-num_classes], x[:, -num_classes:])
    dataset = dataset.map(divide_features_and_labels, num_parallel_calls=AUTOTUNE)
    # define shuffling
    if shuffle:
        dataset = dataset.shuffle(embeddings_and_labels.shape[0])
    # create batches
    dataset = dataset.batch(batch_size)
    # convert to sequence-to-one task if needed
    if type_of_labels == "sequence_to_one":
        def convert_labels_to_sequence_to_one(features, labels):
            tf.print("labels before converting: ", labels, "shape:", tf.shape(labels))
            # argmax to convert to the 2D tensor
            labels = tf.argmax(labels, axis=-1)
            tf.print("labels after argmax: ", labels, "shape:", tf.shape(labels))
            # find a mode for each axis
            labels = tf.map_fn(
                lambda x: tf.unique_with_counts(x).y[tf.argmax(tf.unique_with_counts(x).count, output_type=tf.int32)],
                labels)
            tf.print("labels after converting: ", labels, "shape:", tf.shape(labels))
            labels = tf.cast(labels, tf.float32)
            labels = tf.expand_dims(labels, axis=1)
            return features, labels

        dataset = dataset.map(convert_labels_to_sequence_to_one, num_parallel_calls=AUTOTUNE)
    # apply preprocessing function to images
    if preprocessing_function:
        dataset = dataset.map(lambda x, y: preprocessing_function(x, y), num_parallel_calls=AUTOTUNE)
    # clip values to [0., 1.] if needed
    if clip_values:
        dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)

    # done
    return dataset


if __name__=="__main__":
    input_shape=(10,8)
    window_size=3
    windows_shift=1
    window_stride=1
    num_classes=2

    data=np.ones((10,10))*np.arange(10)[..., np.newaxis]
    data[:,-2:]=0
    data[:5,-2]=1
    data[5:, -1] = 1
    print(data)
    df_columns=["embedding_"+str(i) for i in range(8)]+["label_"+str(i) for i in range(num_classes)]
    df=pd.DataFrame(data,columns=df_columns)
    print(df)

    data_generator=get_tensorflow_sequence_loader(embeddings_and_labels=df, num_classes=num_classes,
                                   batch_size=1, type_of_labels="sequence_to_one",
                                   window_size=window_size, window_shift=windows_shift, window_stride=window_stride,
                                   shuffle= False,
                             preprocessing_function = None,
                             clip_values = None,
                             cache_loaded_seq=None)

    for x,y in data_generator.as_numpy_iterator():
        print(x)
        print(y)
        print("-------------------------------")

    """model=tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64,input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(num_classes,return_sequences=True))
    model.add(tf.keras.layers.Dense(num_classes, activation='relu'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    model.summary()

    model.fit(data_generator, epochs=100)

    for x, y in data_generator.as_numpy_iterator():
        print(y)
        print("prediction")
        print(model.predict(x))
        print("-------------------------------")"""
