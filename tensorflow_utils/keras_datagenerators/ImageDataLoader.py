#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import gc
import multiprocessing
from typing import Optional, Callable, Tuple

from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np

from tensorflow_utils.keras_datagenerators.ImageAugmentor import ImageAugmentor


class ImageDataLoader(Sequence):
    """TODO:write description"""
    horizontal_flip: float
    vertical_flip: float
    horizontal: float
    brightness: float
    shearing: float
    zooming: float
    random_cropping_out: float
    bluring: float
    rotation: float
    scaling: float
    channel_random_noise: float
    worse_quality: float
    mixup: float
    preprocess_function: Callable

    prob_factors_for_each_class: Optional[Tuple[float, ...]]
    paths_with_labels: pd.DataFrame
    batch_size: int
    num_classes:int
    num_workers:int
    pool: multiprocessing.Pool

    def __init__(self, paths_with_labels: pd.DataFrame, batch_size: int, preprocess_function: Optional[Callable] = None,
                 num_classes: Optional[int] = None,
                 horizontal_flip: Optional[float] = None, vertical_flip: Optional[float] = None,
                 shift: Optional[float] = None,
                 brightness: Optional[float] = None, shearing: Optional[float] = None, zooming: Optional[float] = None,
                 random_cropping_out: Optional[float] = None, rotation: Optional[float] = None,
                 scaling: Optional[float] = None,
                 channel_random_noise: Optional[float] = None, bluring: Optional[float] = None,
                 worse_quality: Optional[float] = None,
                 mixup: Optional[float] = None,
                 already_one_hot_encoded:bool = False,
                 prob_factors_for_each_class: Optional[Tuple[float, ...]] = None,
                 pool_workers: int = 4):
        # TODO: write description
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shift = shift
        self.brightness = brightness
        self.shearing = shearing
        self.zooming = zooming
        self.random_cropping_out = random_cropping_out
        self.rotation = rotation
        self.scaling = scaling
        self.channel_random_noise = channel_random_noise
        self.bluring = bluring
        self.worse_quality = worse_quality
        self.mixup = mixup
        self.prob_factors_for_each_class = prob_factors_for_each_class
        self.paths_with_labels = paths_with_labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.already_one_hot_encoded=already_one_hot_encoded
        self.preprocess_function = preprocess_function
        self.num_workers=pool_workers
        # check provided params
        self._check_provided_params()
        # shuffle before start
        self.on_epoch_end()

    def _check_provided_params(self):
        # TODO: write description
        # checking the provided DataFrame
        if self.paths_with_labels.shape[0] == 0:
            raise AttributeError('DataFrame is empty.')
        # check if all provided variables are in the allowed range (usually, from 0..1 or bool)
        if self.horizontal_flip is not None and (self.horizontal_flip < 0 or self.horizontal_flip > 1):
            raise AttributeError('Parameter horizontal_flip should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.vertical_flip is not None and (self.vertical_flip < 0 or self.vertical_flip > 1):
            raise AttributeError('Parameter vertical_flip should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.shift is not None and (self.shift < 0 or self.shift > 1):
            raise AttributeError('Parameter shift should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.brightness is not None and (self.brightness < 0 or self.brightness > 1):
            raise AttributeError('Parameter brightness should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.shearing is not None and (self.shearing < 0 or self.shearing > 1):
            raise AttributeError('Parameter shearing should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.zooming is not None and (self.zooming < 0 or self.zooming > 1):
            raise AttributeError('Parameter zooming should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.random_cropping_out is not None and (self.random_cropping_out < 0 or self.random_cropping_out > 1):
            raise AttributeError('Parameter random_cropping_out should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.rotation is not None and (self.rotation < 0 or self.rotation > 1):
            raise AttributeError('Parameter rotation should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.channel_random_noise is not None and (self.channel_random_noise < 0 or self.channel_random_noise > 1):
            raise AttributeError('Parameter channel_random_noise should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.bluring is not None and (self.bluring < 0 or self.bluring > 1):
            raise AttributeError('Parameter bluring should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.worse_quality is not None and (self.worse_quality < 0 or self.worse_quality > 1):
            raise AttributeError('Parameter worse_quality should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.mixup is not None and (self.mixup < 0 or self.mixup > 1):
            raise AttributeError('Parameter mixup should be float number between 0 and 1, '
                                 'representing the portion of images to be mixup applied.')
        # create a self.pool variable
        self.pool = None
        # calculate the number of classes if it is not provided
        if self.num_classes is None:
            if self.already_one_hot_encoded:
                self.num_classes = self.paths_with_labels.iloc[:,1:].shape[1]
            else:
                self.num_classes = self.paths_with_labels['class'].unique().shape[0]
        # check if provided len of prob_factors_for_each_class is the same as num_classes
        if self.prob_factors_for_each_class is not None:
            if len(self.prob_factors_for_each_class) != self.num_classes:
                raise AttributeError('prob_factors_for_each_class should have num_classes elements. Got %i.' % len(
                    self.prob_factors_for_each_class))
        else:
            # assign every factor to 1
            self.prob_factors_for_each_class = tuple(1. for _ in range(self.num_classes))

    def _create_multiprocessing_pool(self, num_workers:int):
        self.pool = multiprocessing.Pool(num_workers)

    def _realise_multiprocessing_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            del self.pool
        gc.collect()
        self.pool=None


    def _load_and_preprocess_batch(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: write description
        filenames = self.paths_with_labels['filename'].iloc[
                    idx * self.batch_size:(idx + 1) * self.batch_size].values.flatten()

        if self.already_one_hot_encoded:
            labels=self.paths_with_labels.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, 1:].values
        else:
            labels = self.paths_with_labels['class'].iloc[idx * self.batch_size:(idx + 1) * self.batch_size].values.flatten()

        results = []
        for filename_idx in range(filenames.shape[0]):
            label=int(np.argmax(labels[filename_idx])) if self.already_one_hot_encoded else int(labels[filename_idx])
            results.append(self.pool.apply_async(ImageAugmentor.load_and_preprocess_one_image,
                                                 args=(filenames[filename_idx],
                                                       self.horizontal_flip * self.prob_factors_for_each_class[
                                                           label],
                                                       self.vertical_flip * self.prob_factors_for_each_class[
                                                           label],
                                                       self.shift * self.prob_factors_for_each_class[label],
                                                       self.brightness * self.prob_factors_for_each_class[label],
                                                       self.shearing * self.prob_factors_for_each_class[label],
                                                       self.zooming * self.prob_factors_for_each_class[label],
                                                       self.random_cropping_out * self.prob_factors_for_each_class[
                                                           label],
                                                       self.rotation * self.prob_factors_for_each_class[label],
                                                       self.channel_random_noise * self.prob_factors_for_each_class[
                                                           label],
                                                       self.bluring * self.prob_factors_for_each_class[label],
                                                       self.worse_quality * self.prob_factors_for_each_class[
                                                           label])))

        result = []
        for res in results:
            result.append(res.get())
        result = dict(result)

        # create batch output
        image_shape = result[filenames[0]].shape
        data = np.zeros((labels.shape[0],) + image_shape)
        for idx_filename in range(filenames.shape[0]):
            filename = filenames[idx_filename]
            data[idx_filename] = result[filename]

        # one-hot-label encoding
        if not self.already_one_hot_encoded:
            labels = labels.reshape((-1, 1))
            labels = np.eye(self.num_classes)[labels.reshape((-1,)).astype('int32')]

        # mixup
        if self.mixup is not None:
            data, labels = self._mixup(data, labels)

        # clear RAM
        del results, result, filenames,

        return (data.astype('float32'), labels)

    def _mixup(self, images: np.ndarray, labels: np.ndarray, alfa: float = 0.2):
        # TODO: write description
        portion = int(np.ceil(self.mixup * images.shape[0]))
        if portion % 2 == 1: portion += 1
        if portion == 0: return images, labels
        indexes_to_choose = np.random.choice(images.shape[0], portion)
        beta_values = np.random.beta(alfa, alfa, portion // 2)
        # vectorized implementation
        middle_idx = indexes_to_choose.shape[0] // 2
        left_side_images = images[indexes_to_choose[:middle_idx]]
        left_side_labels = labels[indexes_to_choose[:middle_idx]]
        right_side_images = images[indexes_to_choose[middle_idx:]]
        right_side_labels = labels[indexes_to_choose[middle_idx:]]
        # generate new images and labels with betta and (1-betta) coefficients
        new_images_1 = left_side_images * beta_values.reshape((-1,1,1,1)) + right_side_images * (1 - beta_values.reshape((-1,1,1,1)))
        new_images_2 = left_side_images * (1.-beta_values.reshape((-1, 1, 1, 1))) + right_side_images * beta_values.reshape((-1, 1, 1, 1))
        new_labels_1 = left_side_labels * beta_values.reshape((-1,1)) + right_side_labels * (1 - beta_values.reshape((-1,1)))
        new_labels_2 = left_side_labels * (1.-beta_values.reshape((-1, 1))) + right_side_labels *beta_values.reshape((-1, 1))
        # assign generated images back to primary array
        idx_generated_arrays=0
        for idx_indexes_to_choose in range(0,indexes_to_choose.shape[0],2):
            image_idx_1=indexes_to_choose[idx_indexes_to_choose]
            image_idx_2=indexes_to_choose[idx_indexes_to_choose+1]
            images[image_idx_1]=new_images_1[idx_generated_arrays]
            images[image_idx_2]=new_images_2[idx_generated_arrays]
            labels[image_idx_1]=new_labels_1[idx_generated_arrays]
            labels[image_idx_2]=new_labels_2[idx_generated_arrays]
            idx_generated_arrays+=1
        return images, labels

    def on_epoch_end(self):
        # TODO: write description
        # shuffle rows in dataframe
        self.paths_with_labels = self.paths_with_labels.sample(frac=1)
        # clear RAM and realise pools
        if self.pool is not None:
            self._realise_multiprocessing_pool()
        gc.collect()


    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: write description
        # create a multiprocessing pool, if it is None and we call this function first time
        if self.pool is None:
            self._create_multiprocessing_pool(self.num_workers)
        data, labels = self._load_and_preprocess_batch(index)
        if self.preprocess_function is not None:
            data = self.preprocess_function(data)
        return (data, labels)

    def __len__(self) -> int:
        # TODO: write description
        num_steps = int(np.ceil(self.paths_with_labels.shape[0] / self.batch_size))
        return num_steps


    def __del__(self):
        # destructor
        self._realise_multiprocessing_pool()
        del self.paths_with_labels
        gc.collect()
