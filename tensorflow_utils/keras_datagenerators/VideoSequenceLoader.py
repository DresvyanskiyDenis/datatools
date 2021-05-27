#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import multiprocessing
import random
from typing import Optional, Callable, Tuple, List

from scipy.stats import mode
from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np

from tensorflow_utils.keras_datagenerators.ImageAugmentor import ImageAugmentor


class VideoSequenceLoader(Sequence):
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
    num_classes: int
    num_workers: int
    pool: multiprocessing.Pool

    def __init__(self, paths_with_labels: pd.DataFrame,
                 batch_size: int, num_frames_in_seq: int, proportion_of_intersection: float,
                 preprocess_function: Optional[Callable] = None,
                 num_classes: Optional[int] = None,
                 horizontal_flip: Optional[float] = None, vertical_flip: Optional[float] = None,
                 shift: Optional[float] = None,
                 brightness: Optional[float] = None, shearing: Optional[float] = None, zooming: Optional[float] = None,
                 random_cropping_out: Optional[float] = None, rotation: Optional[float] = None,
                 scaling: Optional[float] = None,
                 channel_random_noise: Optional[float] = None, bluring: Optional[float] = None,
                 worse_quality: Optional[float] = None,
                 num_pool_workers: int = 4):
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
        self.paths_with_labels = paths_with_labels
        self.batch_size = batch_size
        self.num_frames_in_seq = num_frames_in_seq
        self.proportion_of_intersection = proportion_of_intersection
        self.num_classes = num_classes
        self.preprocess_function = preprocess_function
        self.num_workers = num_pool_workers
        # check provided params
        self._check_provided_params()
        # prepare dataframe by cutting it on sequences
        self._prepare_dataframe_for_sequence_extraction()
        # shuffle before start
        self.on_epoch_end()

    def _check_provided_params(self):
        # TODO: write description
        # checking the provided DataFrame
        if self.paths_with_labels.columns.to_list() != ['filename', 'frame_num', 'class']:
            raise AttributeError(
                'DataFrame columns should be \'filename\', \'frame_num\', \'class\'. Got %s.' % self.paths_with_labels.columns)
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
        if self.num_frames_in_seq <= 0 or type(self.num_frames_in_seq) != int:
            raise AttributeError('Parameter num_frames_in_seq should be positive integer number.')
        if self.proportion_of_intersection < 0 or self.proportion_of_intersection >= 1:
            raise AttributeError('Parameter proportion_of_intersection should be float number between 0 and 1.')
        # create a pool of workers to do multiprocessing during loading and preprocessing
        self.pool = multiprocessing.Pool(self.num_workers)
        # calculate the number of classes if it is not provided
        if self.num_classes is None:
            self.num_classes = self.paths_with_labels['class'].unique().shape[0]

    def _prepare_dataframe_for_sequence_extraction(self) -> None:
        # calculate step of window based on proportion
        step = int(np.ceil(self.num_frames_in_seq * self.proportion_of_intersection))
        # sort dataframe by filename and then by frame_num
        self.paths_with_labels = self.paths_with_labels.sort_values(by=['filename', 'frame_num'])
        # divide dataframe on sequences
        self.sequences = self._divide_dataframe_on_sequences(self.paths_with_labels, self.num_frames_in_seq, step)

    def _divide_dataframe_on_sequences(self, dataframe: pd.DataFrame, seq_length: int, step: int) -> List[pd.DataFrame]:
        unique_filenames = np.unique(dataframe['filename'])
        # TODO: implement dividing whole dataframe on sequences
        sequences = []
        for unique_filename in unique_filenames:
            df_to_cut = dataframe[dataframe['filename'] == unique_filename]
            try:
                cut_df = self._divide_dataframe_on_list_of_seq(df_to_cut, seq_length, step)
            except AttributeError:
                continue
            sequences = sequences + cut_df
        return sequences

    def _divide_dataframe_on_list_of_seq(self, dataframe: pd.DataFrame, seq_length: int, step: int) -> List[
        pd.DataFrame]:
        sequences = []
        if dataframe.shape[0] < seq_length:
            # TODO: create your own exception
            raise AttributeError('The length of dataframe is less than seq_length. '
                                 'Dataframe length:%i, seq_length:%i' % (dataframe.shape[0], seq_length))
        num_sequences = int(np.ceil((dataframe.shape[0] - seq_length) / step + 1))
        for num_seq in range(num_sequences - 1):
            start = num_seq * step
            end = start + seq_length
            seq = dataframe.iloc[start:end]
            sequences.append(seq)
        # last sequence is from the end of sequence to end-seq_length
        sequences.append(dataframe.iloc[(dataframe.shape[0] - seq_length):])
        return sequences

    def _load_and_preprocess_batch(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: implement it
        sequences_to_load = self.sequences[idx * self.batch_size:(idx + 1) * self.batch_size]
        # concatenate selected sequences to load them
        concatenated_df = pd.concat(sequences_to_load, ignore_index=True)
        # form labels
        labels = np.array(concatenated_df['class']).reshape((-1, self.num_frames_in_seq))
        # form filenames for loading images
        filenames = concatenated_df['filename'] + '_'+concatenated_df['frame_num'].astype('str')+'.jpg'
        # load and augment if needed all images by filenames using multiprocessing
        processes = []
        for filename in filenames:
            processes.append(self.pool.apply_async(ImageAugmentor.load_and_preprocess_one_image,
                                                   args=(filename,
                                                         self.horizontal_flip,
                                                         self.vertical_flip,
                                                         self.shift,
                                                         self.brightness,
                                                         self.shearing,
                                                         self.zooming,
                                                         self.random_cropping_out,
                                                         self.rotation,
                                                         self.channel_random_noise,
                                                         self.bluring,
                                                         self.worse_quality)
                                                   ))
        # get results of multiprocessing
        results = []
        for process in processes:
            results.append(process.get())
        # now it is in dict[filename->image]
        results = dict(results)
        # relocate images from dict to numpy array
        image_shape = results[filenames[0]].shape
        image_sequences = np.zeros((self.batch_size, self.num_frames_in_seq) + image_shape, dtype='uint8')
        filenames = np.array(filenames).reshape((-1, self.num_frames_in_seq))
        for batch_idx in range(filenames.shape[0]):
            for frame_idx in range(filenames.shape[1]):
                image_sequences[batch_idx, frame_idx] = results[filenames[batch_idx, frame_idx]]
        # apply preprocess function if needed
        if self.preprocess_function is not None:
            image_sequences = image_sequences.reshape((-1,) + image_shape)
            image_sequences = self.preprocess_function(image_sequences)
            image_sequences = image_sequences.reshape((self.batch_size, self.num_frames_in_seq) + image_shape)
        # clear RAM
        del results
        return image_sequences, labels

    def _one_hot_encoding(self, labels: np.ndarray) -> np.ndarray:
        # one-hot-label encoding
        labels = np.eye(self.num_classes)[labels.reshape((-1,)).astype('int32')]
        return labels

    def _sequence_to_one_transformation(self, labels:np.ndarray)->np.ndarray:
        labels=mode(labels, axis=1)[0].reshape((-1,1))
        return labels

    def on_epoch_end(self):
        # shuffle sequences
        random.shuffle(self.sequences)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        batch_image_sequences, labels = self._load_and_preprocess_batch(index)
        # turn into sequence-to-one labeling
        labels=self._sequence_to_one_transformation(labels)
        # one-hot encoding
        labels = self._one_hot_encoding(labels)
        return batch_image_sequences, labels

    def __len__(self) -> int:
        num_batches = int(np.ceil(len(self.sequences) / self.batch_size))
        return num_batches
