#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains ChunksGenerator_preprocessing class. The description of the class is below.

"""
import random
from multiprocessing.pool import Pool
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import tensorflow as tf

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from sklearn.preprocessing import StandardScaler

from preprocessing.data_preprocessing.audio_preprocessing_utils import cut_data_on_chunks, \
    extract_opensmile_features_from_audio_sequence, extract_mfcc_from_audio_sequence, \
    extract_subwindow_EGEMAPS_from_audio_sequence, extract_HLDs_from_LLDs, extract_combined_features_with_sibwindows

Data_type_format = Dict[str, Tuple[np.ndarray, int]]
data_preprocessing_types = ('raw', 'LLD', 'HLD', 'EGEMAPS', 'MFCC', 'HLD_EGEMAPS')
labels_types = ('sequence_to_one',)


class ChunksGenerator_preprocessing(tf.keras.utils.Sequence):
    """tf.keras.utils.Sequence generator to pass it in model.fit() function.
    However, it can be used as simple generator as well.
    The generator takes data and labels in dict[str, np.ndarray] format
    (str - filename, np.ndarray - corresponding audio sequence or labels)
    The data will be cut onto fixed number of chunks same length according to article:

    "An Efficient Temporal Modeling Approach for Speech Emotion Recognition byMapping Varied Duration Sentences
    into Fixed Number of Chunks"
    https://indico2.conference4me.psnc.pl/event/35/contributions/3415/attachments/531/557/Wed-1-9-1.pdf

    Please, read article for the better understanding what is going on here :)

    """
    num_chunks: int  # the fixed number of chunks on which audio will be cut (it is calculated inside class).
    window_length: float  # the length of window (chunk)
    data: Data_type_format  # data in dict format
    data_preprocessing_mode: Optional[str]  # the preprocessing mode: do we want to calculate LLDs, HLDs, EGEMAPS, MFCC
    # combine them or just use raw audio
    labels: Dict[str, np.ndarray]  # supplied labels in dict format
    labels_type: str  # defines the type of labels. Can be sequence-to-one and sequence-to-sequence
    num_mfcc: Optional[int]  # if data_preprocessing mode is MFCC, defines the number of MFCC to be extracted
    normalization: bool  # should supplied batch be normalized or not
    subwindow_size: Optional[float]  # if data_preprocessing_mode is HLD, EGEMAPS or HLD_EGEMAPS, defined subwindows,
    # onto which chunk will be cutted to calculate functionals within subwindow
    subwindow_step: Optional[float]  # the step of subwindow
    precutting: bool  # if audio sequence is too long, it can be cut onto several pieces, and every piece will be
    # considered then as separate instance
    precutting_window_size: Optional[float]  # the size of these pieces
    precutting_window_step: Optional[float]  # the step on which window will be shifted in cutting process

    def __init__(self, *, sequence_max_length: float, window_length: float, data: Optional[Data_type_format] = None,
                 data_preprocessing_mode: Optional[str] = 'raw', num_mfcc: Optional[int] = 128,
                 labels: Dict[str, np.ndarray] = None, labels_type: str = 'sequence_to_one', batch_size: int = 32,
                 normalization: bool = False, normalizer:Optional[object]=None,
                 one_hot_labeling: Optional[bool] = None, num_classes: Optional[int] = None,
                 subwindow_size: Optional[float] = None, subwindow_step: Optional[float] = None,
                 precutting: bool = False,
                 precutting_window_size: Optional[float] = None, precutting_window_step: Optional[float] = None):
        """Assigns basic values for data cutting and providing, loads labels, defines how data will be loaded
            and check if all the provided values are in appropriate format

        :param sequence_max_length: float
                    max length of sequence in seconds. Can be decimal.
        :param window_length: float
                    the length of window for data cutting, in seconds. Can be decimal.
        :param data: Optional[Data_type_format=Dict[str, Tuple[np.ndarray, int]]]
                    if load_mode is 'data', then the data in almost the same format as labels must be
                    provided (besides np.ndarray it contains a sample_rate int number also)
        :param data_preprocessing_mode: str
                    can be one of ('raw', 'LLD', 'HLD','EGEMAPS', 'MFCC')
        :param labels: Dict[str, np.ndarray]
                    labels for data. dictionary represents mapping str -> np.ndarray, where
                    str denotes filename and np.ndarray denotes label on each timestep, or 1 label per whole filename,
                    thus, shape of np.ndarray will be (1,)
        :param batch_size: int

        :param normalization: bool
                    Apply normalization to the features before yielding or not.
        :param normalizer: object
                    if not None, the provided normalizer will be applied to transform data with the help
                    of .transform() function. These objects should be sklearn.preprocessing objects.
                    like: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        :param one_hot_labeling: Optional[bool]
                    Apply one hot labeling to the output labels or not.
        :param num_classes: Optional[int]
                    if one_hot_labeling is True, supplies the number of classes to generate one-hot label
        :param subwindow_size: Optional[float]
                    if data_preprocessing_mode equals either 'HLD' or 'EGEMAPS', then
                    specifies the size of subwindows to extract high-level descriptors or EGEMAPS from each subwindow
                    For example, if we have the cut chunk of data with shape (num_timesteps, num_features), then it
                    will be transformed/recomputed to (num_windows, num_defined_HLDs*num_features), e. g. chunk of data with
                    shape (100, 10) and subwindow_size = 0.1, subwindow_step=0.05 and (max, min, mean, std) functionals
                    will be transformed to (19, 40) array.
        :param subwindow_step: Optional[float]
                    if data_preprocessing_mode equals either 'HLD' or 'EGEMAPS', then
                    specifies the step of subwindows.
        :param precutting: bool
                    if need, pre-cut sequence on windows, which will be used used then separate as instances of batch
                    every window will be processed as earlies (cut on fixed number of chunks, transformed to defined
                    features)
        :param precutting_window_size: Optional[float]
                    if precutting=True, defines the size of window, which will be applied to cut original sequence on
                    slices. Float value in seconds
        :param precutting_window_step: Optional[float]
                    if precutting=True, defines the step of window, which will be applied to cut original sequence on
                    slices. Float value in seconds
        """
        # params assigning
        self.num_chunks = int(np.ceil(sequence_max_length / window_length))
        self.window_length = window_length
        self.batch_size = batch_size
        self.num_mfcc = num_mfcc
        self.normalization = normalization
        self.normalizer=normalizer
        self.one_hot_labeling = one_hot_labeling
        self.num_classes = num_classes
        self.subwindow_size = subwindow_size
        self.subwindow_step = subwindow_step
        self.precutting = precutting
        self.precutting_window_size = precutting_window_size
        self.precutting_window_step = precutting_window_step

        # check if data_preprocessing_mode has an appropriate value
        if data_preprocessing_mode in data_preprocessing_types:
            self.data_preprocessing_mode = data_preprocessing_mode
        else:
            raise AttributeError(
                'data_preprocessing_mode can be either \'raw\', \'LLD\', \'MFCC\', \'EGEMAPS\' or \'HLD\'. Got %s.' % (
                    data_preprocessing_mode))

        # check if data has an appropriate format
        if isinstance(data, dict):
            self.data = data
            # precut data if need
            if self.precutting:
                self.data = self.precut_sequence_on_slices(self.data)
            # cut provided data
            self.data = self._cut_data_in_dict(self.data)
            # preprocess data
            self._preprocess_all_data()
        else:
            raise AttributeError('With \'data\' load mode the data should be in dict format. Got %s.' % (type(data)))

        # check if labels are provided in an appropriate way
        if isinstance(labels, dict):
            if len(labels.keys()) == 0:
                raise AttributeError('Provided labels are empty.')
            elif not isinstance(list(labels.keys())[0], str):
                raise AttributeError('Labels should be a dictionary in str->np.ndarray format.')
            elif not isinstance(list(labels.values())[0], np.ndarray):
                raise AttributeError('Labels should be a dictionary in str->np.ndarray format.')
            else:
                self.labels = labels
        else:
            raise AttributeError('Labels should be a dictionary in str->np.ndarray format.')
        # check if labels_type is provided in appropriate way
        if labels_type in labels_types:
            self.labels_type = labels_type
        else:
            raise AttributeError(
                'Labels_type can be either \'sequence_to_one\' or \'sequence_to_sequence\'. Got %s.' % (labels_type))

        # check if one_hot_labeling and num_classes are proveded either both or not
        if one_hot_labeling:
            if num_classes == None:
                raise AttributeError(
                    'If one_hot_labeling=True, the number of classes should be provided. Got %i.' % (num_classes))

        # form indexes for batching then
        self.indexes = self._form_indexes()
        # shuffle data
        self.on_epoch_end()

    def __len__(self) -> int:
        """Calculates how many batches per one epoch will be.

        :return: int
                how many batches will be per epoch.
        """
        num_batches = int(np.ceil(self._calculate_overall_size_of_data() / self.batch_size))
        return num_batches

    def __getitem__(self, index):
        """yield batch of instances.

        :param index: int
                the index of the batch
        :return: Tuple[np.ndarray, np.ndarray]
                the batch of x,y pairs
        """
        data, labels = self._form_batch_with_data_load_mode(index)
        if self.one_hot_labeling:
            labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        return data.astype('float32'), labels

    def on_epoch_end(self):
        """Do some actions at the end of epoch.
           We use random.shuffle function to shuffle list of indexes presented via self.indexes
        :return: None
        """
        random.shuffle(self.indexes)

    """
    def normalize_batch_of_chunks(self, batch_of_chunks: np.ndarray) -> np.ndarray:
        \"""Normalizes a batch of chunks via StandardScaler() normalizer.
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        :param batch_of_chunks: np.ndarray
                    ndarray with shape (batch_size, num_chunks, window_size, num_features)
        :return: np.ndarray
                    ndarray with the same shape as batch_of_chunks
        \"""
        # reshape array to 2D (window_num, num_features)
        array_to_normalize = batch_of_chunks.reshape((-1, batch_of_chunks.shape[-1]))
        # create scaler
        normalizer = StandardScaler()
        # fit and transform data
        array_to_normalize = normalizer.fit_transform(array_to_normalize)
        # reshape obtained data back to initial shape
        result_array = array_to_normalize.reshape(batch_of_chunks.shape)
        return result_array
        """

    def _form_batch_with_data_load_mode(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forms batch, which consists of data and labels chosen according index from shuffled self.indexes.
           The data and labels slice depends on index and defined as index*self.batch_size:(index+1)*self.batch_size
           Thus, index represents the number of slice.

        :param index: int
                    the number of slice
        :return: Tuple[np.ndarray, np.ndarray]
                    data and labels slice
        """
        # select batch_size indexes from randomly shuffled self.indexes
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size if (index + 1) * self.batch_size <= len(self.indexes) \
            else len(self.indexes)
        indexes_to_load = self.indexes[start_idx:end_idx]
        # form already loaded data and labels in batch
        batch_data = []
        batch_labels = []
        for chosen_idx in indexes_to_load:
            filename, chunk_idx = chosen_idx
            # self.data[filename][0][np.newaxis,...] :
            # > self.data[filename] provides us (np.ndarray, sample_rate) according to filename
            # > self.data[filename][0] provides us np.ndarray with shape (num_chunks, window_size, num_features)
            # > self.data[filename][0][np.newaxis,...] expand array to add new axis for success concatenation further
            current_data = self.data[filename][0][chunk_idx][np.newaxis, ...]
            current_labels = self.labels[filename][np.newaxis, ...]
            batch_data.append(current_data)
            batch_labels.append(current_labels)
        # concatenate extracted data and labels
        batch_data = np.concatenate(batch_data, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)
        batch_labels = batch_labels.reshape(batch_labels.shape[:2])
        return batch_data, batch_labels

    def _preprocess_raw_audio(self, raw_audio: np.ndarray, sample_rate: int,
                              preprocess_type: str, num_mfcc: Optional[int] = None) -> np.ndarray:
        """Preprocesses raw audio with 1 channel according chosen preprocess_type

        :param raw_audio: np.ndarray
                    raw audio in numpy format. Can be with shape (n_samples,) or (n_samples,1)
        :param sample_rate: int
                    sample rate of audio
        :param preprocess_type: str
                    which algorithm to choose to preprocess? The types are presented in variable data_preprocessing_types
                    at the start of the script (where imports)
        :param num_mfcc: int
                    if preprocess_type=='MFCC', defines how much mfcc are needed.
        :return: np.ndarray
                    extracted features from audio
        """
        # check if raw_audio have 2 dimensions
        if len(raw_audio.shape) == 1:
            raw_audio = raw_audio[..., np.newaxis]
        elif len(raw_audio.shape) != 2:
            raise AttributeError('raw_audio should be 1- or 2-dimensional. Got %i dimensions.' % (len(raw_audio.shape)))

        if preprocess_type == 'raw':
            preprocessed_audio = raw_audio
        elif preprocess_type == 'LLD':
            preprocessed_audio = extract_opensmile_features_from_audio_sequence(raw_audio, sample_rate, preprocess_type)
        elif preprocess_type == 'MFCC':
            preprocessed_audio = extract_mfcc_from_audio_sequence(raw_audio.astype('float32'), sample_rate, num_mfcc)
        elif preprocess_type == 'EGEMAPS':
            preprocessed_audio = extract_subwindow_EGEMAPS_from_audio_sequence(raw_audio, sample_rate,
                                                                               subwindow_size=self.subwindow_size,
                                                                               subwindow_step=self.subwindow_step)
        elif preprocess_type == 'HLD':
            preprocessed_audio = extract_opensmile_features_from_audio_sequence(raw_audio, sample_rate, 'LLD')
            preprocessed_audio = extract_HLDs_from_LLDs(preprocessed_audio, window_size=self.subwindow_size,
                                                        window_step=self.subwindow_step,
                                                        required_HLDs=('min', 'max', 'mean', 'std'))
        elif preprocess_type == 'HLD_EGEMAPS':
            preprocessed_audio = extract_combined_features_with_sibwindows(raw_audio, sample_rate,
                                                                           self.subwindow_size, self.subwindow_step,
                                                                           feature_types=('HLD', 'EGEMAPS'))
        else:
            raise AttributeError(
                'preprocess_type should be either \'LLD\', \'MFCC\' or \'EGEMAPS\'. Got %s.' % (preprocess_type))
        return preprocessed_audio

    def _preprocess_cut_audio(self, cut_audio: np.ndarray, sample_rate: int,
                              preprocess_type: str, num_mfcc: Optional[int] = None, filename: Optional[str] = None) -> \
            Union[np.ndarray, Tuple[np.ndarray, str]]:
        """ Extract defined in preprocess_type features from cut audio.

        :param cut_audio: np.ndarray
                    cut audio with shape (num_chunks, window_length, 1) or (num_chunks, window_length)
        :param sample_rate: int
                    sample rate of audio
        :param preprocess_type: str
                    which algorithm to choose to preprocess? The types are presented in variable data_preprocessing_types
                    at the start of the script (where imports)
        :param num_mfcc: int
                    if preprocess_type=='MFCC', defines how much mfcc are needed.
        :param filename: str
                    filename of audio, uses for multiprocessing, to get the result
        :return: np.ndarray or (np.ndarray, str)
                    features extracted from each window. The output shape is (num_chunks, window_length, num_features)
                    for multiprocessing returns the filename as well.
        """
        batches = []
        for batch_idx in range(cut_audio.shape[0]):
            chunks = []
            for chunk_idx in range(cut_audio.shape[1]):
                extracted_features = self._preprocess_raw_audio(cut_audio[batch_idx, chunk_idx], sample_rate,
                                                                preprocess_type, num_mfcc)
                chunks.append(extracted_features[np.newaxis, ...])
            chunks = np.concatenate(chunks, axis=0)[np.newaxis, ...]
            batches.append(chunks)
        batches = np.concatenate(batches, axis=0)
        if filename != None:
            return batches, filename
        return batches

    def _preprocess_all_data(self):
        """Preprocesses all data located in self.data variable
            The function uses multiprocessing. This accelarates the preprocessing procedure
            (how much: depends on the CPU).
        :return: None
        """
        # function uses multiprocessing
        params_for_pool = []

        for key, value in self.data.items():
            cut_sequence, sample_rate = value
            filename = key
            # multiprocessing for preprocessing data
            params_for_pool.append((cut_sequence, sample_rate, self.data_preprocessing_mode, self.num_mfcc, filename))

        with Pool(processes=8) as pool:
            results = pool.starmap(self._preprocess_cut_audio, params_for_pool)

        for items in results:
            extracted_features, filename = items
            cut_sequence, sample_rate = self.data[filename]
            self.data[filename] = (extracted_features, sample_rate)

        if self.normalization:
            if self.normalizer is None:
                self.data, self.normalizer= self._normalize_all_data(self.data, return_normalizer=True)
            else:
                self.data = self._normalize_all_data(self.data, normalizer=self.normalizer)

    def _normalize_all_data(self, data:Data_type_format, return_normalizer:bool=False,
                        normalizer:Optional[object]=None) -> Union[Data_type_format, Tuple[Data_type_format, object]]:
        if normalizer is None:
            normalizer=StandardScaler()
            concatenated_data = []
            for key, item in data.items():
                array, sample_rate = item
                concatenated_data.append(array)
            concatenated_data = np.concatenate(concatenated_data, axis=0)
            normalizer=normalizer.fit(concatenated_data.reshape((-1, concatenated_data.shape[-1])))
            del concatenated_data
        for key, item in data.items():
            array, sample_rate = item
            old_shape=array.shape
            array=array.reshape((-1, array.shape[-1]))
            array=normalizer.transform(array)
            array=array.reshape(old_shape)
            data[key]=(array, sample_rate)
        if return_normalizer:
            return data, normalizer
        return data

    def _form_indexes(self) -> List[Union[int, Tuple[str, int]]]:
        """Forms random indexes depend on data type was loaded.

        :return: indexes
                list[str] or list[list[str, int]], see the explanation above.
        """
        # if data_mode=='path', we simply make permutation of indexes of self.data_filenames
        indexes = []
        for key, value in self.data.items():
            # extract values
            data_array, sample_rate = value
            filename = key
            # make permutation for np.ndarray of particular filename
            num_indexes = data_array.shape[0]
            permutations = np.random.permutation(num_indexes)
            # append indexes randomly permutated
            for i in range(num_indexes):
                indexes.append((filename, permutations[i]))
        return indexes

    def _calculate_overall_size_of_data(self) -> int:
        """Calculates the overall size of data.
           It is assumed that self.data is in the format dict(str, np.ndarray), where
           np.ndarray has shape (Num_batches, num_chunks, window_size, num_features)
                    num_chunks is evaluated in self.__init__() function
                    window_size is evaluated in self.__init__() function
                    num_features is provided by data initially supplied
            The reason that data has 4 dimensions is that it was cut on slices

        :return: int
            the overall size of data, evaluated across all entries in dict
        """
        sum = 0
        for key, value in self.data.items():
            sum += value[0].shape[0]
        return sum

    def _cut_sequence_on_slices(self, sequence: np.ndarray, sample_rate: int) -> np.ndarray:
        """cut provided sequence on fixed number of slices. The cutting process carries out on axis=0
           The number of chunks is fixed and equals self.num_chunks.
           For example, 2-d sequence becomes an 3-d
           (num_timesteps, num_features) -> (num_chunks, window_size, num_features)

        :param sequence: np.ndarray
                    all dimensions starting from 2 is supported.
        :param sample_rate: int
                    sample rate of sequence
        :return: np.ndarray
                    cut on chunks array
        """
        # evaluate window length and step in terms of units (indexes of arrays).
        # self.window_length is presented initially  in seconds
        window_length_in_units = int(self.window_length * sample_rate)
        window_step_in_units = int(np.ceil((sequence.shape[0] - window_length_in_units) / (self.num_chunks - 1)))
        # cut data with special function in audio_preprocessing_utils.py
        cut_data = cut_data_on_chunks(data=sequence, chunk_length=window_length_in_units,
                                      window_step=window_step_in_units)
        # check if we got as much chunks as we wanted
        if len(cut_data) != self.num_chunks:
            raise ValueError('Function _cut_sequence_on_slices(). The number of cut chunks is not the same as '
                             'was computed in __init__() function. cut_data.shape[0]=%i, should be: %i'
                             % (len(cut_data), self.num_chunks))
        # concatenate cut chunks in np.ndarray
        cut_data = np.array(cut_data)
        return cut_data

    def _cut_data_in_dict(self, data: Data_type_format) -> Data_type_format:
        """Cuts every instance in dict (self.data) onto slices (chunks).

        :param data: Dict[str, Tuple[np.ndarray, int]]
        :return: Dict[str, Tuple[np.ndarray, int]]
        """
        for key, value in data.items():
            batches, sample_rate = value
            if self.precutting == False:
                batches = batches[np.newaxis, ...]
            batches_array = []
            for batch_idx in range(batches.shape[0]):
                array = self._cut_sequence_on_slices(batches[batch_idx], sample_rate)
                batches_array.append(array[np.newaxis, ...])
            batches_array = np.concatenate(batches_array, axis=0)
            data[key] = (batches_array, sample_rate)
        return data

    def precut_sequence_on_slices(self, data: Data_type_format) -> Data_type_format:
        """ Cuts self.data if self.precutting==True. It is needed, if audio sequence is too long
        and we want to divide it onto several parts.

        :param data: Dict[str, Tuple[np.ndarray, int]]
        :return: Dict[str, Tuple[np.ndarray, int]]
        """
        for key, value in data.items():
            array, sample_rate = value
            window_size_in_units = int(round(sample_rate * self.precutting_window_size))
            window_step_in_units = int(round(sample_rate * self.precutting_window_step))
            array = cut_data_on_chunks(data=array, chunk_length=window_size_in_units,
                                       window_step=window_step_in_units)
            array = [array[i][np.newaxis, ...] for i in range(len(array))]
            array = np.concatenate(array, axis=0)
            data[key] = (array, sample_rate)
        return data

    def get_dict_predictions(self, model:tf.keras.Model):
        predictions={}
        for key, item in self.data.items():
            array, sample_rate = item
            prediction=model.predict(array)
            predictions[key]=prediction
        return predictions

    def __get_features_shape__(self):
        """Provides the shape of features, e.g
        (num_chunks, window_size, num_features) will give (window_size, num_features), since it is the features
        with timesteps. It is the input shape, which will be passed into model.

        :return: Tuple[int,...]
                The shape of features
        """
        # get first arbitrary key
        key = list(self.data.keys())[0]
        # return shape of data
        return self.data[key][0].shape




if __name__ == "__main__":
    pass
