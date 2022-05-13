#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils for extracting deep embeddings from provided data by neural networks (tensorflow)

Module contains functions for deep embeddings extraction by provided extractor, which is tensorflow model.


List of functions:

    * extract_deep_embeddings_from_batch_of_images - extracts deep embeddings from provided batch of images (in np.ndarray form)
    * extract_deep_embeddings_from_images_in_dir - extracts deep embeddings from all images located in provided directory
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


import gc
import os
from typing import Optional, Tuple, Callable
import tensorflow as tf
import numpy as np
import pandas as pd
from preprocessing.data_preprocessing.image_preprocessing_utils import load_batch_of_images


def extract_deep_embeddings_from_batch_of_images(images:np.ndarray, extractor:tf.keras.Model, batch_size:Optional[int]=None)->np.ndarray:
    """Extracts deep embeddings from provided batch of images via using extractor

    :param images: np.ndarray
            batch of images in np.ndarray form
    :param extractor: tf.keras.Model
            neural network model, implemented in tensorflow
    :param batch_size: Optional[int]
            batch size for deep embeddings extraction (in case if overall number of images is high and you want
            to split calculation into several mini-batches). It is needed, when the whole batch of images cannot be fit into
            model for one step.
    :return: np.ndarray
            extracted embeddings with shape [images.shape[0], num_embeddings]
    """
    if batch_size is None:
        batch_size=images.shape[0]
    embeddings=extractor.predict(images, batch_size=batch_size)
    return embeddings

def extract_deep_embeddings_from_images_in_dir(path_to_dir:str, extractor:tf.keras.Model,
                                               return_type:str='df', batch_size:int=16,
                                               preprocessing_functions:Tuple[Callable[[np.ndarray], np.ndarray], ...]=None)->pd.DataFrame:
    """Extracts deep embeddings from all images in provided directory. Returns pd.DataFrame with full paths to each image and
       deep embeddings.

    :param path_to_dir: str
            path to directory with images
    :param extractor: tf.keras.Model
            neural network model, implemented in tensorflow
    :param return_type: str
            can be only df. Other types are not supported yet.
    :param batch_size: int
            Parameter for model, indicated how much images per step it should process.
    :param preprocessing_functions: Tuple[Callable[[np.ndarray], np.ndarray], ...]
            Tuple of the functions for preprocessing data before extraction deep embeddings. It is needed in case if
            data should be preprocessed before fit into model.
    :return: pd.DataFrame
            DataFrame with absolute paths and deep embeddings for every image
    """
    if return_type not in ('df'):
        raise AttributeError('return type can be only \'df\'. Got %s.'%return_type)
    # load all filenames
    filenames=np.array(os.listdir(path_to_dir))
    # define columns for df
    num_embeddings = tuple(extractor.output_shape)[1]
    columns=['filename'] + [str(i) for i in range(num_embeddings)]
    # create dataframe for saving features
    embeddings=pd.DataFrame(columns=columns)
    # load batch_size images and then predict them
    for filename_idx in range(0, filenames.shape[0], batch_size):
        batch_filenames=filenames[filename_idx:(filename_idx+batch_size)]
        batch_filenames=tuple(os.path.join(path_to_dir, filename) for filename in batch_filenames)
        loaded_images=load_batch_of_images(batch_filenames)
        # preprocess loaded image if needed
        if preprocessing_functions is not None:
            for preprocessing_function in preprocessing_functions:
                loaded_images=preprocessing_function(loaded_images)
        # extract embeddings
        extracted_emb=extract_deep_embeddings_from_batch_of_images(loaded_images, extractor, batch_size)
        pd_to_concat=pd.DataFrame(data=np.concatenate([np.array(batch_filenames).reshape((-1,1)),
                                                      extracted_emb], axis=1),
                                                    columns=columns)
        embeddings=embeddings.append(pd_to_concat, ignore_index=True)
        del loaded_images
        gc.collect()
    return embeddings


def extract_deep_embeddings_from_images_in_df(paths_to_images:pd.DataFrame, extractor:tf.keras.Model, output_dir:str,
                                              batch_size: int = 16,
                                              preprocessing_functions: Tuple[Callable[[np.ndarray], np.ndarray], ...] = None,
                                              include_labels:bool =False
                                              ) ->None:
    """ Extracts deep embeddings from the images using provided extractor (tf Model).
        Images should be passed as a pd.DataFrame, where the first and single columns should provide the full path to the image.
        All extracted deep embeddings will be written to as a separate csv file in the output_dir.

    :param paths_to_images: pd.DataFrame
            DataFrame with a single column with full paths to the images
    :param extractor: tf.keras.Model
            Model to extract deep embeddings
    :param output_dir: str
            path to save extracted deep embeddings
    :param batch_size: int
            batch size for the extractor
    :param preprocessing_functions: Tuple[Callable[[np.ndarray], np.ndarray], ...]
            Tuple of Callable functions to apply to the images before extracting deep embeddings.
    :param include_labels: bool
            If True, labels will be included in the output csv file
    :param num_labels: int
            Number of labels to include in the output csv file
    """
    # check if output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # form the final dataframe, which will be written
    # define columns for df
    num_embeddings = tuple(extractor.output_shape)[1]
    columns = ['filename'] + ["embedding_"+str(i) for i in range(num_embeddings)]
    if include_labels:
        columns += ["label_"+str(i) for i in range(paths_to_images.shape[1]-1)]
    # create dataframe for saving features
    extracted_deep_embeddings = pd.DataFrame(columns=columns)
    # save the "template" csv file to append to it in future
    extracted_deep_embeddings.to_csv(os.path.join(output_dir, 'extracted_deep_embeddings.csv'), index=False)

    # load batch_size images and then predict them
    for extraction_idx, filename_idx in enumerate(range(0, paths_to_images.shape[0], batch_size)):
        batch_filenames = paths_to_images.iloc[filename_idx:(filename_idx + batch_size),0].values.flatten()
        # load images and convert them to the float32 type for the extractor
        loaded_images = load_batch_of_images(batch_filenames)
        loaded_images=loaded_images.astype(np.float32)
        # preprocess loaded image if needed
        if preprocessing_functions is not None:
            for preprocessing_function in preprocessing_functions:
                loaded_images = preprocessing_function(loaded_images)
        # extract embeddings
        extracted_emb = extract_deep_embeddings_from_batch_of_images(loaded_images, extractor, batch_size)
        # if we want to include labels in the resulting csv file
        if include_labels:
            labels = paths_to_images.iloc[filename_idx:(filename_idx + batch_size),1:].values
            extracted_emb = np.concatenate([extracted_emb, labels], axis=1)
        # append the filenames as a first column
        extracted_emb = pd.DataFrame(data=np.concatenate([np.array(batch_filenames).reshape((-1, 1)),
                                                          extracted_emb], axis=1),
                                     columns=columns)
        # append them to the already extracted ones
        extracted_deep_embeddings = extracted_deep_embeddings.append(extracted_emb, ignore_index=True)
        # dump the extracted data to the file
        if extraction_idx % 1000 == 0:
            extracted_deep_embeddings.to_csv(os.path.join(output_dir, 'extracted_deep_embeddings.csv'), index=False,
                                             header=False, mode="a")
            # clear RAM
            extracted_deep_embeddings=pd.DataFrame(columns=columns)
            gc.collect()
        del loaded_images
        gc.collect()
    # dump remaining data to the file
    extracted_deep_embeddings.to_csv(os.path.join(output_dir, 'extracted_deep_embeddings.csv'), index=False,
                                     header=False, mode="a")
