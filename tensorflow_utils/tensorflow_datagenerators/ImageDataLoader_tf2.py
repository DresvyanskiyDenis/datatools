from functools import partial
from typing import Tuple, Optional, List, Callable

import pandas as pd
import tensorflow as tf

from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_augmentations import \
    random_rotate90_image, random_flip_vertical_image, random_flip_horizontal_image
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_image_VGGFace2

Tensorflow_Callable = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]

def load_image(path_to_image, label):
    # read the image from disk, decode it, convert the data type to
    # floating point
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image, label)

def get_tensorflow_image_loader(paths_and_labels: pd.DataFrame, batch_size: int, augmentation: bool = False,
                             augmentation_methods: Optional[List[Tensorflow_Callable]] = None,
                             preprocessing_function: Optional[Tensorflow_Callable] = None,
                             clip_values: Optional[bool] = None,
                             cache_loaded_images:Optional[bool]=None) -> tf.data.Dataset:
    """TODO: write description

    :param paths_and_labels:
    :param batch_size:
    :param augmentation:
    :param augmentation_prob:
    :param augmentation_methods:
    :param preprocessing_function:
    :param clip_values:
    :return:
    """
    AUTOTUNE = tf.data.AUTOTUNE
    # create tf.data.Dataset from provided paths to the images and labels
    dataset = tf.data.Dataset.from_tensor_slices((paths_and_labels.iloc[:, 0], paths_and_labels.iloc[:, 1:]))
    # define shuffling
    dataset = dataset.shuffle(paths_and_labels.shape[0])
    # define image loading function
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    # cache for better performance if specified
    if cache_loaded_images:
        dataset = dataset.cache()
    # augment if needed
    if augmentation:
        # go through all augmentation methods and "roll the dice (probability)" every time before applying the
        # specific augmentation
        for method in augmentation_methods:
            dataset = dataset.map(lambda x, y: method(x,y),
                                  num_parallel_calls=AUTOTUNE)
    # create batches
    dataset = dataset.batch(batch_size)
    # apply preprocessing function to images
    if preprocessing_function:
        dataset = dataset.map(lambda x, y: preprocessing_function(x, y))
    # clip values to [0., 1.] if needed
    if clip_values:
        dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))
    dataset = dataset.prefetch(AUTOTUNE)

    # done
    return dataset


if __name__ == '__main__':
    pass
