import tensorflow as tf
import numpy as np
from preprocessing.data_normalizing_utils import VGGFace2_normalization
from preprocessing.data_preprocessing.image_preprocessing_utils import load_image
import matplotlib.pyplot as plt

from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_augmentations import random_rotate90_image, \
    random_crop_image, random_change_brightness_image, \
    random_change_contrast_image, random_change_saturation_image, random_worse_quality_image, \
    random_convert_to_grayscale_image
from tensorflow_utils.tensorflow_datagenerators.tensorflow_image_preprocessing import preprocess_image_VGGFace2


def main():
    path=r"E:\test\data\002_2016-03-17_Paris\Expert_video\frame_25375.png"
    image=load_image(path)
    img_tf_1 = tf.Variable(image, dtype=tf.float32)
    img_tf_2 = tf.Variable(image, dtype=tf.float32)
    img_tf_3 = tf.Variable(image, dtype=tf.float32)
    img_tf_4 = tf.Variable(image, dtype=tf.float32)
    img_tf_5 = tf.Variable(image, dtype=tf.float32)
    img_tf_6 = tf.Variable(image, dtype=tf.float32)

    img_tf_concat=tf.concat([tf.expand_dims(img_tf_1, axis=0),tf.expand_dims(img_tf_2, axis=0),
                             tf.expand_dims(img_tf_3, axis=0),tf.expand_dims(img_tf_4, axis=0),
                             tf.expand_dims(img_tf_5, axis=0),tf.expand_dims(img_tf_6, axis=0)], axis=0)
    tf.print(img_tf_1)
    #img_tf_concat, _ = augmentation(img_tf_concat, tf.constant(2.), 1.0)
    img_tf_1, _ = random_worse_quality_image(img_tf_1, 0, 1.)
    tf.print(img_tf_1)
    img_tf_2, _ = random_worse_quality_image(img_tf_2, 0, 1.)
    img_tf_3, _ = random_worse_quality_image(img_tf_3, 0, 1.)
    img_tf_4, _ = random_worse_quality_image(img_tf_4, 0, 1.)
    img_tf_5, _ = random_worse_quality_image(img_tf_5, 0, 1.)


    """img_tf_4 = tf.Variable(image)
    img_tf_4 = augmentation(img_tf_4, tf.constant(0.1))

    img_tf_5 = tf.Variable(image)
    img_tf_5 = augmentation(img_tf_5, tf.constant(0.1))

    img_tf_6 = tf.Variable(image)
    img_tf_6 = augmentation(img_tf_6, tf.constant(0.1))

    img_tf_7 = tf.Variable(image)
    img_tf_7 = augmentation(img_tf_7, tf.constant(0.1))"""



    fig = plt.figure()
    fig.add_subplot(2,4,1)
    plt.imshow(image)
    fig.add_subplot(2,4,2)
    plt.imshow(tf.cast(img_tf_1, dtype=tf.uint8))
    fig.add_subplot(2,4,3)
    plt.imshow(tf.cast(img_tf_2, dtype=tf.uint8))
    fig.add_subplot(2,4,4)
    plt.imshow(tf.cast(img_tf_3, dtype=tf.uint8))
    fig.add_subplot(2,4,5)
    plt.imshow(tf.cast(img_tf_4, dtype=tf.uint8))
    fig.add_subplot(2,4,6)
    plt.imshow(tf.cast(img_tf_5, dtype=tf.uint8))
    fig.add_subplot(2,4,7)
    #plt.imshow(img_tf_concat[5])
    #fig.add_subplot(2,4,8)

    """fig.add_subplot(2,4,5)
    plt.imshow(img_tf_4)
    fig.add_subplot(2,4,6)
    plt.imshow(img_tf_5)
    fig.add_subplot(2,4,7)
    plt.imshow(img_tf_6)
    fig.add_subplot(2,4,8)
    plt.imshow(img_tf_7)"""

    plt.show()


if __name__ == '__main__':
    main()