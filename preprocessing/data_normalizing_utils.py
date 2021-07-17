#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains different methods to normalize and scale data
Uses sklearn library.

List of functions:
    * get_trained_minmax_scaler - provides trained on supplied data MinMaxScaler from sklearn library
    * transform_data_with_scaler - transforms supplied data with provided scaler
    * normalize_min_max_data - normalizes/transforms supplied data using MinMax normalization and provides trained
    scaler, if it is needed.
    * z_normalization - apply z-normalization to data
    * l2_normalization - applies l2-normalization to data
    * power_normalization - applies power-normalization to data (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
    * image_scaling_to_unit_range - scales pixels to range [0,1]
    * VGGFace2_normalization - applies normalization, used in VggFace2 work
"""
from typing import Tuple, Optional, Union
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, normalize, StandardScaler

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


vgg_face2_mean = (91.4953, 103.8827, 131.0912)

def get_trained_minmax_scaler(data:np.ndarray, feature_range:Tuple[float, float]=(-1,1)) -> object:
    """Trains and returns MinMaxScaler from sklearn library.

    :param data: np.ndarray
            data, on which scaler will be learnt
    :param feature_range: tuple(int, int)
            range of the future features
    :return: sklearn.preprocessing.MinMaxScaler
            trained on data scaler
    """
    normalizer = MinMaxScaler(feature_range=feature_range)
    normalizer = normalizer.fit(data)
    return normalizer

def transform_data_with_scaler(data:np.ndarray, scaler:object) -> np.ndarray:
    """Transforms data by passed scaler object (from sklearn.preprocessing).

    :param data: np.ndarray
            data to trasform
    :param scaler: sklearn.preprocessing object
            scaler, which will apply transformation operation to data
    :return: np.ndarray
            transformed data
    """
    transformed_data=scaler.transform(data)
    return transformed_data

def normalize_min_max_data(data:np.ndarray, return_scaler:bool=False) -> Tuple[np.ndarray,Optional[object]] or np.ndarray:
    """Normalize data via minmax normalization with the help of sklearn.preprocessing.MinMaxScaler.
       Normalization will use last dimension.

    :param data: numpy.ndarray
                data to normalize
    :param return_scaler: bool
                return MinMaxScaler object, if you need it for further using
    :return: (numpy.ndarray, object) or numpy.ndarray
                return either data or data with scaler
    """
    normalizer=get_trained_minmax_scaler(data)
    transformed_data=transform_data_with_scaler(data, normalizer)
    if return_scaler:
        return transformed_data, normalizer
    else:
        return transformed_data

def z_normalization(data:np.ndarray, return_scaler:bool=False,
                    scaler:Optional[object]=None)->Union[np.ndarray, Tuple[np.ndarray, object]]:
    """Normalizes data with provided normalizer, or generates new one and fit it.
    Uses sklearn.preprocessing.StandardScaler() object.

    :param data: np.ndarray
                data in 2d-format (n_samples, n_features) or 1d-format (n_samples,)
    :param return_scaler: bool
                Should function return fitted scaler or not.
    :param scaler: object (sklearn.preprocessing.StandardScaler)
                If not None, the provided scaler will be used as normalizer.
    :return: np.ndarray or (np.ndarray, scaler)
                If return_scaler==False, returns normalized data
                else returns normalized data and fitted scaler
    """
    # check if data is in appropriate format
    if len(data.shape)>2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.'%(len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape)==1:
        data=data[..., np.newaxis]

    # if no scaler supplied, create scaler and fit it
    if scaler is None:
        scaler=StandardScaler()
        scaler.fit(data)
    # transform data
    data=scaler.transform(data)
    # return scaler if need
    if return_scaler:
        return data, scaler
    return data

def l2_normalization(data:np.ndarray)->np.ndarray:
    """Normalizes data with l2 normalization. Each instance (vector) will be normalized independently.
        Uses sklearn.preprocessing.normalize() function.

    :param data: np.ndarray
                data in 2d-format (n_samples, n_features) or 1d-format (n_samples,)
    :return: np.ndarray
                normalized data
    """
    # check if data is in appropriate format
    if len(data.shape) > 2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.' % (len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape) == 1:
        data = data[..., np.newaxis]
    # normalize data. axis=1 means that each instance (row) will be independently normalized.
    data=normalize(data, axis=1)
    return data


def power_normalization(data:np.ndarray, return_scaler:bool=False,
                        scaler:Optional[object]=None)->Union[np.ndarray, Tuple[np.ndarray, object]]:
    """Normalizes provided data via power normalization.
    More: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    Uses sklearn.preprocessing.PowerTransformer() object.
    :param data: np.ndarray
                data in 2d-format (n_samples, n_features) or 1d-format (n_samples,)
    :param return_scaler: bool
                Should function return fitted scaler or not.
    :param scaler: object (sklearn.preprocessing.StandardScaler)
                If not None, the provided scaler will be used as normalizer.
    :return: np.ndarray or (np.ndarray, scaler)
                If return_scaler==False, returns normalized data
                else returns normalized data and fitted scaler
    """
    # check if data is in appropriate format
    if len(data.shape) > 2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.' % (len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape) == 1:
        data = data[..., np.newaxis]

    # if no scaler supplied, create scaler and fit it
    if scaler is None:
        scaler = PowerTransformer()
        scaler.fit(data)
    # transform data
    data = scaler.transform(data)
    # return scaler if need
    if return_scaler:
        return data, scaler
    return data

def image_scaling_to_unit_range(img:np.ndarray)->np.ndarray:
    """Scales pixel values to range [0, 1]

    :param img: np.ndarray
            3D array, which represents image.
    :return: np.ndarray
            Scaled array.
    """
    return img/255.

def VGGFace2_normalization(img:np.ndarray)->np.ndarray:
    """Applies normalization to the image, as it did in https://arxiv.org/abs/1710.08092
    https://github.com/WeidiXie/Keras-VGGFace2-ResNet50

    :param img: np.ndarray
            3D array, which represents image.
    :return: np.ndarray
            Normalized array
    """
    img = img[:, :, ::-1] - vgg_face2_mean
    return img
