import os
from functools import partial
from typing import Callable, List, Dict, Union, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class TemporalDataLoader(Dataset):

    def __init__(self, paths_with_labels:Dict[str, pd.DataFrame], label_columns:List[str],
                 window_size:Union[int, float], stride:Union[int, float],
                 consider_timestamps:Optional[bool]=False,
                 preprocessing_functions:List[Callable]=None,
                 augmentation_functions:Dict[Callable, float]=None, shuffle:bool=False):
        self.paths_with_labels = paths_with_labels
        self.label_columns = label_columns
        self.window_size = window_size
        self.stride = stride
        self.consider_timestamps = consider_timestamps
        self.preprocessing_functions = preprocessing_functions
        self.augmentation_functions = augmentation_functions
        self.shuffle = shuffle

        # check provided variables
        if consider_timestamps and not isinstance(window_size, float):
            raise ValueError("If consider_timestamps is True, window_size should be float.")
        if consider_timestamps and not isinstance(stride, float):
            raise ValueError("If consider_timestamps is True, stride should be float.")

        # cut all data on windows
        self.__cut_all_data_on_windows()

        # to get item every time in the __getitem__ method, we need to create a list of pointers.
        # every pointer points to a concrete window located in self.cut_windows
        # in this way, we can easily shuffle them and get the access to the windows really quickly
        self.pointers = []
        for key, windows in self.cut_windows.items():
            for window in windows:
                self.pointers.append((key, window))
        # shuffle pointers if needed
        if self.shuffle:
            np.random.shuffle(self.pointers)



    def __len__(self):
        # len of the dataset is the sum of all windows, but we can easily get it from the pointers
        return len(self.pointers)

    def __getitem__(self, idx):
        # get the data and labels using pointer
        _, window = self.pointers[idx]
        paths = window['path'].values
        labels = window[self.label_columns].values
        # load images (they are loaded using torch build-in function)
        images = [read_image(path) for path in paths]
        # turn grey image into RGB if needed
        images = [image if image.shape[0] == 3 else torch.cat([image, image, image]) for image in images]
        # augment images if needed
        if self.augmentation_functions is not None:
            images = [self.__augment(image) for image in images]
        # preprocess images if needed
        if self.preprocessing_functions is not None:
            images = [self.__preprocess_image(image) for image in images]
        # stack images into one tensor. The output shape is (seq_len, channels, height, width)
        images = torch.stack(images)
        # change type to float32
        images = images.type(torch.float32)
        # turn labels into tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        return images, labels

    def get_sequence_length(self):
        """ Returns the length of the sequence. """
        return self.pointers[0][1].shape[0]


    def __preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        for func in self.preprocessing_functions:
            image = func(image)
        return image

    def __augment(self, image: torch.Tensor) -> torch.Tensor:
        for func, prob in self.augmentation_functions.items():
            if torch.rand(1) < prob:
                image = func(image)
        return image


    def __cut_all_data_on_windows(self):
        """ Cuts all data on windows. """
        self.cut_windows = {}
        for key, frames in self.paths_with_labels.items():
            self.cut_windows[key] = self.__create_windows_out_of_frames(frames, self.window_size, self.stride)
        # check if there were some sequences with not enough frames to create a window
        # they have been returned as None, so we need to remove them
        self.cut_windows = {key: windows for key, windows in self.cut_windows.items() if windows is not None}



    def __create_windows_out_of_frames(self, frames:pd.DataFrame, window_size:Union[int, float], stride:Union[int, float])\
            ->Union[List[pd.DataFrame],None]:
        """ Creates windows of frames out of a pd.DataFrame with frames. Each window is a pd.DataFrame with frames.
        The columns are the same as in the original pd.DataFrame.

        :param frames: pd.DataFrame
                pd.DataFrame with frames. Columns format: ['path', ..., 'label_0', ..., 'label_n']
        :param window_size: Union[int, float]
                Size of the window. If int, it is the number of frames in the window. If float, it is the time in seconds.
        :param stride: Union[int, float]
                Stride of the window. If int, it is the number of frames in the window. If float, it is the time in seconds.
        :return:
        """
        # calculate the number of frames in the window
        if self.consider_timestamps:
            timestep = min(frames['timestamp'].iloc[1]-frames['timestamp'].iloc[0], frames['timestamp'].iloc[-1]-frames['timestamp'].iloc[-2])
            num_frames = int(np.round(window_size / timestep))
            # stride in seconds needs to be converted to number of frames
            stride = int(np.round(stride / timestep))
        else:
            num_frames = window_size
        # create windows
        windows = self.__cut_sequence_on_windows(frames, window_size=num_frames, stride=stride)

        return windows

    def __cut_sequence_on_windows(self, sequence:pd.DataFrame, window_size:int, stride:int)->Union[List[pd.DataFrame],None]:
        """ Cuts one sequence of values (represented as pd.DataFrame) into windows with fixed size. The stride is used
        to move the window. If there is not enough values to fill the last window, the window starting from
        sequence_end-window_size is added as a last window.

        :param sequence: pd.DataFrame
                Sequence of values represented as pd.DataFrame
        :param window_size: int
                Size of the window in number of values/frames
        :param stride: int
                Stride of the window in number of values/frames
        :return: List[pd.DataFrame]
                List of windows represented as pd.DataFrames
        """
        # check if the sequence is long enough
        # if not, return None and this sequence will be skipped in the __cut_all_data_on_windows method
        if sequence.shape[0] < window_size:
            return None
        windows = []
        # cut sequence on windows using while and shifting the window every step
        window_start = 0
        window_end = window_start + window_size
        while window_end <= len(sequence):
            windows.append(sequence.iloc[window_start:window_end])
            window_start += stride
            window_end += stride
        # add last window if there is not enough values to fill it
        if window_start < len(sequence):
            windows.append(sequence.iloc[-window_size:])
        return windows


