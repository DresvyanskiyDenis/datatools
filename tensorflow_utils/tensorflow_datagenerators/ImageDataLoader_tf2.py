from typing import Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
import os

from src.NoXi.preprocessing.data_preprocessing import generate_rel_paths_to_images_in_all_dirs
from src.NoXi.preprocessing.labels_preprocessing import combine_path_to_images_with_labels_many_videos, \
    generate_paths_to_labels, load_all_labels_by_paths


def load_and_preprocess_data(path_to_data: str, path_to_labels: str, frame_step: int) ->  pd.DataFrame:
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
        new_key=key.split(os.path.sep)[-2]+str(os.path.sep)
        new_key=new_key+'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key+'novice'
        train_labels[new_key]=train_labels.pop(key)
    for key in list(dev_labels.keys()):
        new_key=key.split(os.path.sep)[-2]+str(os.path.sep)
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        dev_labels[new_key]=dev_labels.pop(key)
    for key in list(test_labels.keys()):
        new_key=key.split(os.path.sep)[-2]+str(os.path.sep)
        new_key = new_key + 'expert' if 'expert' in key.split(os.path.sep)[-1] else new_key + 'novice'
        test_labels[new_key]=test_labels.pop(key)
    # combine paths to images (data) with labels
    train_image_paths_and_labels = combine_path_to_images_with_labels_many_videos(paths_with_images=paths_to_images,
                                                                                  labels=train_labels,
                                                                                  sample_rate_annotations=25,
                                                                                  frame_step=frame_step)
    # shuffle train data
    train_image_paths_and_labels = train_image_paths_and_labels.sample(frac=1).reset_index(drop=True)
    # create abs path for all paths instead of relative (needed for generator)
    train_image_paths_and_labels['filename']=train_image_paths_and_labels['filename'].apply(lambda x:os.path.join(path_to_data, x))
    # done
    return train_image_paths_and_labels


def load_image(path_to_image, label):
    # read the image from disk, decode it, convert the data type to
    # floating point
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_png(image, channels=3)
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    return (image, label)

def data_load():
    # loading data
    frame_step = 5
    path_to_data = r"C:\Users\Dresvyanskiy\Desktop\data"
    # french data
    path_to_labels_french = r"C:\Users\Dresvyanskiy\Desktop\NoXi_annotations_reliable_gold_standard_classification_with_additional_train_data\French"
    train_french = load_and_preprocess_data(path_to_data, path_to_labels_french,
                                                                     frame_step)
    train = train_french
    return train


def create_model(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,5,input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(64,3,input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(128,3,input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(128,3,input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    return model

def main():
    AUTOTUNE=tf.data.AUTOTUNE
    train = data_load()
    print(train)

    dataset = tf.data.Dataset.from_tensor_slices((train.iloc[:,0], train.iloc[:,1:]))
    dataset = dataset.shuffle(train.shape[0])
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(AUTOTUNE)


    model = create_model((224,224,3))
    model.compile(optimizer='SGD', loss='categorical_crossentropy')
    model.summary()

    model.fit(x=dataset, epochs=10)

if __name__ == '__main__':
    main()