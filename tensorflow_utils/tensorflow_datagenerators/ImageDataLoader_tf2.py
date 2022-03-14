from functools import partial
from typing import Tuple, Optional, List, Callable

import tensorflow as tf
import pandas as pd
import numpy as np
import os

from src.NoXi.preprocessing.data_preprocessing import generate_rel_paths_to_images_in_all_dirs
from src.NoXi.preprocessing.labels_preprocessing import combine_path_to_images_with_labels_many_videos, \
    generate_paths_to_labels, load_all_labels_by_paths
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_augmentations import \
    random_rotate90_image, random_flip_vertical_image, random_flip_horizontal_image, random_crop_image, \
    random_change_brightness_image, random_change_contrast_image, random_change_saturation_image, \
    random_worse_quality_image, random_convert_to_grayscale_image
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_image_VGGFace2

Tensorflow_Callable = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def load_and_preprocess_data(path_to_data: str, path_to_labels: str, frame_step: int) -> pd.DataFrame:
    """

    :param path_to_data:
    :param path_to_labels:
    :return:
    """
    # generate paths to images (data)
    paths_to_images = generate_rel_paths_to_images_in_all_dirs(path_to_data, image_format="png")
    # generate paths to train/dev/test labels
    paths_train_labels = generate_paths_to_labels(os.path.join(path_to_labels, "train"))
    paths_dev_labels = generate_paths_to_labels(os.path.join(path_to_labels, "dev"))
    paths_test_labels = generate_paths_to_labels(os.path.join(path_to_labels, "test"))
    # load labels
    train_labels = load_all_labels_by_paths(paths_train_labels)
    dev_labels = load_all_labels_by_paths(paths_dev_labels)
    test_labels = load_all_labels_by_paths(paths_test_labels)
    del paths_train_labels, paths_dev_labels, paths_test_labels
    # change the keys of train_labels/dev_labels/test_labels to have only the name with pattern name_of_video/novice_or_expert
    for key in list(train_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + str(os.path.sep)
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        train_labels[new_key] = train_labels.pop(key)
    for key in list(dev_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + str(os.path.sep)
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        dev_labels[new_key] = dev_labels.pop(key)
    for key in list(test_labels.keys()):
        new_key = key.split(os.path.sep)[-2] + str(os.path.sep)
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        test_labels[new_key] = test_labels.pop(key)
    # combine paths to images (data) with labels
    train_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                  labels=train_labels,
                                                                                  sample_rate_annotations=25,
                                                                                  frame_step=frame_step)
    # shuffle train data
    train_image_paths_and_labels = train_image_paths_and_labels.sample(frac=1).reset_index(drop=True)
    # create abs path for all paths instead of relative (needed for generator)
    train_image_paths_and_labels['filename'] = train_image_paths_and_labels['filename'].apply(
        lambda x: os.path.join(path_to_data, x))
    # done
    return train_image_paths_and_labels


def load_image(path_to_image, label):
    # read the image from disk, decode it, convert the data type to
    # floating point
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image, label)


def data_load():
    # loading data
    frame_step = 5
    path_to_data = r"C:\Users\Professional\Desktop\test\data"
    # french data
    path_to_labels_french = r"C:\Users\Professional\Desktop\test\NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data\French"
    train_french = load_and_preprocess_data(path_to_data, path_to_labels_french,
                                            frame_step)
    train = train_french
    return train


def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 5, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(64, 3, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(128, 3, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(128, 3, input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    return model


def roll_prob_dice(barrier):
    dice = tf.random.uniform([], 0., 1.)
    cond = tf.math.less(dice, barrier)
    return cond


def augment(x, y, augmentation_function):
    result = tf.cond(tf.math.less(tf.random.uniform([], 0., 1.), 0.1),
                     lambda: augmentation_function(x, y),
                     lambda: x, y)
    return result


def get_tensorflow_generator(paths_and_labels: pd.DataFrame, batch_size: int, augmentation: bool = False,
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



def func_test_1():
    # params
    augment = True
    augment_prob = 0.05
    augment_methods = [
        partial(random_rotate90_image, probability=augment_prob),
        partial(random_flip_vertical_image, probability=augment_prob),
        partial(random_flip_horizontal_image, probability=augment_prob),
        partial(random_crop_image, probability=augment_prob),
        partial(random_change_brightness_image, probability=augment_prob, min_max_delta=0.35),
        partial(random_change_contrast_image, probability=augment_prob, min_factor=0.5, max_factor=1.5),
        partial(random_change_saturation_image, probability=augment_prob, min_factor=0.5, max_factor=1.5),
        partial(random_worse_quality_image, probability=augment_prob, min_factor=25, max_factor=99),
        partial(random_convert_to_grayscale_image, probability=augment_prob)
    ]
    preprocessing_function = None
    clip_value = False
    batch_size = 8

    train = data_load()
    print(train)

    dataset = get_tensorflow_generator(train, batch_size, augment, augment_prob, augment_methods,
                                       preprocessing_function,
                                       clip_value)
    import matplotlib.pyplot as plt
    for x,y in dataset:
        tf.print(x[0].dtype)
        tf.print(x[0])
        fig = plt.figure()
        fig.add_subplot(2, 4, 1)
        plt.imshow(tf.cast(x[0], dtype=tf.int32))
        fig.add_subplot(2, 4, 2)
        plt.imshow(tf.cast(x[1], dtype=tf.int32))
        fig.add_subplot(2, 4, 3)
        plt.imshow(tf.cast(x[2], dtype=tf.int32))
        fig.add_subplot(2, 4, 4)
        plt.imshow(tf.cast(x[3], dtype=tf.int32))
        fig.add_subplot(2, 4, 5)
        plt.imshow(tf.cast(x[4], dtype=tf.int32))
        fig.add_subplot(2, 4, 6)
        plt.imshow(tf.cast(x[5], dtype=tf.int32))
        fig.add_subplot(2, 4, 7)
        plt.imshow(tf.cast(x[6], dtype=tf.int32))
        fig.add_subplot(2, 4, 8)
        plt.imshow(tf.cast(x[7], dtype=tf.int32))

        plt.show()



def main():
    # params
    augment = True
    augment_prob = 0.1
    augment_methods = [
        partial(random_rotate90_image, probability=augment_prob),
        partial(random_flip_vertical_image, probability=augment_prob),
        partial(random_flip_horizontal_image, probability=augment_prob),
        #random_crop_image,
        #partial(random_change_brightness_image, min_max_delta=0.35),
        #partial(random_change_contrast_image, min_factor=0.5, max_factor=1.5),
        #partial(random_change_saturation_image, min_factor=0.5, max_factor=1.5),
        #partial(random_worse_quality_image, min_factor=25, max_factor=99),
        #random_convert_to_grayscale_image
    ]
    preprocessing_function = preprocess_image_VGGFace2
    clip_value = False
    batch_size = 32

    train = data_load()
    print(train)

    dataset = get_tensorflow_generator(train, batch_size, augment, augment_prob, augment_methods,
                                       preprocessing_function,
                                       clip_value)

    model = create_model((224, 224, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy')
    model.summary()

    model.fit(x=dataset, epochs=10)


if __name__ == '__main__':
    func_test_1()
