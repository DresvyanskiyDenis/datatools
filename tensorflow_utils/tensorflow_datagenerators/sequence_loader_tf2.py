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
        # function to convert labels to sequence-to-one option, where for one window only one label is related
        def convert_labels_to_sequence_to_one(features, labels):
            # sum the labels through the 1 dimension
            labels = tf.reduce_sum(labels, axis=1)
            # normalize labels
            labels=tf.linalg.normalize(labels, ord=1, axis=-1)[0]
            labels = tf.cast(labels, tf.float32)
            return features, labels
        # apply constructed function to the labels
        dataset = dataset.map(convert_labels_to_sequence_to_one, num_parallel_calls=AUTOTUNE)
    # apply preprocessing function to images if needed
    if preprocessing_function:
        dataset = dataset.map(lambda x, y: preprocessing_function(x, y), num_parallel_calls=AUTOTUNE)
    # clip values to [0., 1.] if needed
    if clip_values:
        dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=AUTOTUNE)
    # cache for better performance if specified
    if cache_loaded_seq:
        dataset = dataset.cache()
    # prefetch
    dataset = dataset.prefetch(AUTOTUNE)

    # done
    return dataset


if __name__=="__main__":
    # tests for sequence loader, especially sequence-to-one option
    num_features=3
    num_instances=21
    window_size=3
    windows_shift=1
    window_stride=1
    num_classes=3
    input_shape = (window_size, num_features)

    data=np.ones((num_instances,num_features+num_classes))*np.arange(num_instances)[..., np.newaxis]
    data[:,-num_classes:]=0
    data[:num_instances//3,-3]=1
    data[num_instances//3:num_instances//3*2, -2] = 1
    data[num_instances // 3 * 2:, -1] = 1
    # change labels for testing
    data[0, -num_classes:] = np.array([0, 0, 1]) # 0
    data[1, -num_classes:] = np.array([0, 0, 1])  # 1
    data[2, -num_classes:] = np.array([1, 0, 0])  # 2
    data[3, -num_classes:] = np.array([1, 0, 0])  # 3
    data[4, -num_classes:] = np.array([0, 1, 0])  # 4
    data[5, -num_classes:] = np.array([0, 1, 0])  # 5
    data[6, -num_classes:] = np.array([0, 1, 0])  # 6
    data[7, -num_classes:] = np.array([0, 0, 1])  # 7
    data[8, -num_classes:] = np.array([0, 0, 1])  # 8
    data[9, -num_classes:] = np.array([0, 0, 1])  # 9
    data[10, -num_classes:] = np.array([1, 0, 0])  # 10
    data[11, -num_classes:] = np.array([1, 0, 0])  # 11
    data[12, -num_classes:] = np.array([1, 0, 0])  # 12
    data[13, -num_classes:] = np.array([0, 1, 0])  # 13
    data[14, -num_classes:] = np.array([0, 1, 0])  # 14
    data[15, -num_classes:] = np.array([1, 0, 0])  # 15
    data[16, -num_classes:] = np.array([0, 0, 1])  # 16
    data[17, -num_classes:] = np.array([0, 1, 0])  # 17
    data[18, -num_classes:] = np.array([0, 0, 1])  # 18
    data[19, -num_classes:] = np.array([0, 0, 1])  # 19
    data[20, -num_classes:] = np.array([1, 0, 0])  # 20

    print(data)
    df_columns=["embedding_"+str(i) for i in range(num_features)]+["label_"+str(i) for i in range(num_classes)]
    df=pd.DataFrame(data,columns=df_columns)
    print(df)

    data_generator=get_tensorflow_sequence_loader(embeddings_and_labels=df, num_classes=num_classes,
                                   batch_size=1, type_of_labels="sequence_to_one",
                                   window_size=window_size, window_shift=windows_shift, window_stride=window_stride,
                                   shuffle= False,
                             preprocessing_function = None,
                             clip_values = None,
                             cache_loaded_seq=None)

    for i, (x,y) in enumerate(data_generator.as_numpy_iterator()):
        #print(x)
        print("%i:"%i,y.shape)
        #print("-------------------------------")

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
