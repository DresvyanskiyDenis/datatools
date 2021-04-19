"""
TODO: write description of package
"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras_datagenerators.ImageDataLoader import ImageDataLoader

"""
TODO: write description of module
"""
import math
import shutil
import time
from typing import Optional, Tuple, Dict, NamedTuple

import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf

"""from preprocessing.data_preprocessing.image_preprocessing_utils import load_image, save_image, resize_image
from preprocessing.face_recognition_utils import recognize_the_most_confident_person_retinaFace, \
    extract_face_according_bbox, load_and_prepare_detector_retinaFace"""

class Label(NamedTuple):
    # TODO: write description
    boredom: int
    engagement: int
    confusion:int
    frustration:int

def sort_images_according_their_class(path_to_images:str, output_path:str, path_to_labels:str):
    dict_labels=load_labels_to_dict(path_to_labels)
    dirs_with_images=os.listdir(path_to_images)
    # check if output path is existed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    num_classes=np.unique(np.array(list(x.engagement for x in dict_labels.values()))).shape[0]
    # create subdirectories for classes
    for num_class in range(num_classes):
        if not os.path.exists(os.path.join(output_path, str(num_class))):
            os.makedirs(os.path.join(output_path, str(num_class)), exist_ok=True)
    # copy images according their class
    for dir_with_images in dirs_with_images:
        if not dir_with_images in dict_labels.keys():
            continue
        class_num=dict_labels[dir_with_images].engagement
        image_filenames=os.listdir(os.path.join(path_to_images, dir_with_images))
        for image_filename in image_filenames:
            shutil.copy(os.path.join(path_to_images, dir_with_images, image_filename),
                    os.path.join(output_path, str(class_num), image_filename))





def load_labels_to_dict(path:str)->Dict[str, Label]:
    # TODO:write description
    labels_df=pd.read_csv(path)
    labels_df['ClipID']=labels_df['ClipID'].apply(lambda x: x.split('.')[0])
    #labels_df.columns=[labels_df.columns[0]]+[x.lower() for x in labels_df.columns[1:]]
    labels_dict=dict(
        (row[1].iloc[0],
         Label(*row[1].iloc[1:].values))
        for row in labels_df.iterrows()
    )
    return labels_dict




def form_dataframe_of_relative_paths_to_data_with_labels(path_to_data:str, labels_dict:Dict[str,Label])-> pd.DataFrame:
    # TODO: write description
    directories_according_path=os.listdir(path_to_data)
    df_with_relative_paths_and_labels=pd.DataFrame(columns=['filename','class'])
    for dir in directories_according_path:
        if not dir in labels_dict.keys():
            continue
        img_filenames=os.listdir(os.path.join(path_to_data, dir))
        img_filenames=[os.path.join(dir, x) for x in img_filenames]
        label=labels_dict[dir].engagement
        labels=[label for _ in range(len(img_filenames))]
        tmp_df=pd.DataFrame(data=np.array([img_filenames, labels]).T, columns=['filename', 'class'])
        df_with_relative_paths_and_labels=df_with_relative_paths_and_labels.append(tmp_df)
    return df_with_relative_paths_and_labels


def tmp_model()->tf.keras.Model:
    model_tmp=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),include_top=False,
                                                             weights='imagenet', pooling='avg')
    x=tf.keras.layers.Dense(512, activation="relu")(model_tmp.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    result_model=tf.keras.Model(inputs=model_tmp.inputs, outputs=[x])
    return result_model







if __name__ == '__main__':
    # params
    path_to_train_frames=r'E:\Databases\DAiSEE\DAiSEE\train_preprocessed\sorted_faces'
    path_to_train_labels=r'E:\Databases\DAiSEE\DAiSEE\Labels\TrainLabels.csv'
    path_to_dev_frames=r'E:\Databases\DAiSEE\DAiSEE\dev_preprocessed\sorted_faces'
    path_to_dev_labels=r'E:\Databases\DAiSEE\DAiSEE\Labels\ValidationLabels.csv'
    '''output_path=r'D:\Databases\DAiSEE\dev_preprocessed\sorted_faces'
    sort_images_according_their_class(path_to_images=path_to_dev_frames, output_path=output_path,
                                      path_to_labels=path_to_dev_labels)'''
    input_shape=(224,224,3)
    batch_size=32
    paths_with_labels=pd.DataFrame(columns=['filename', 'class'])
    filenames=[os.path.join(path_to_train_frames, "1",filename) for filename
               in os.listdir(os.path.join(path_to_train_frames, "1"))]
    labels=[1 for _ in range(len(filenames))]
    filenames=filenames+[os.path.join(path_to_train_frames, "0",filename) for filename
                         in os.listdir(os.path.join(path_to_train_frames, "0"))]
    labels=labels+[0 for _ in range(len(os.listdir(os.path.join(path_to_train_frames, "0"))))]
    filenames=np.array(filenames).reshape((-1,1))
    labels = np.array(labels).reshape((-1, 1))
    paths_with_labels=paths_with_labels.append(pd.DataFrame(np.concatenate([filenames, labels], axis=-1),
                                                            columns=['filename', 'class']))
    paths_with_labels['filename']=paths_with_labels['filename'].astype('str')
    paths_with_labels['class'] = paths_with_labels['class'].astype('int32')
    generator=ImageDataLoader(paths_with_labels=paths_with_labels, batch_size=batch_size, preprocess_function=None,
                 horizontal_flip = None, vertical_flip = None,
                 shift= None,
                 brightness= None, shearing= 0.5, zooming = None,
                 random_cropping_out= None, rotation= None,
                 scaling= None,
                 channel_random_noise = None, bluring= None,
                 worse_quality= None,
                 mixup= None)

    for x,y in generator:
        print(x.shape, y.shape)
        a=1+2



