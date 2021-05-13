#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""
import gc
import os
from typing import Optional, Tuple, Callable

import tensorflow as tf
import numpy as np
import pandas as pd
from preprocessing.data_preprocessing.image_preprocessing_utils import load_batch_of_images


def extract_deep_embeddings_from_batch_of_images(images:np.ndarray, extractor:tf.keras.Model, batch_size:Optional[int]=None)->np.ndarray:
    # TODO: write description
    if batch_size is None:
        batch_size=images.shape[0]
    embeddings=extractor.predict(images, batch_size=batch_size)
    return embeddings

def extract_deep_embeddings_from_images_in_dir(path_to_dir:str, extractor:tf.keras.Model,
                                               return_type:str='df', batch_size:int=16,
                                               preprocessing_functions:Tuple[Callable[[np.ndarray], np.ndarray], ...]=None)->pd.DataFrame:
    # TODO: write description
    if return_type not in ('df','dict'):
        raise AttributeError('return type can be only \'dict\' or \'df\'. Got %s.'%return_type)
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



if __name__=='__main__':
    pass