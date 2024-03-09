from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TemporalEmbeddingsLoader(Dataset):

    def __init__(self, embeddings_with_labels:Dict[str, pd.DataFrame], label_columns:List[str],
                 feature_columns:List[str],
                 window_size:Union[int, float], stride:Union[int, float],
                 consider_timestamps:Optional[bool]=False,
                 only_consecutive_windows:Optional[bool]=True,
                 preprocessing_functions:List[Callable]=None, shuffle:bool=False):
        super().__init__()
        self.embeddings_with_labels = embeddings_with_labels
        self.label_columns = label_columns
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.stride = stride
        self.consider_timestamps = consider_timestamps
        self.only_consecutive_windows = only_consecutive_windows
        self.preprocessing_functions = preprocessing_functions
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
            for idx_window, window in enumerate(windows):
                self.pointers.append((key+f"_{idx_window}", window))
        # shuffle pointers if needed
        if self.shuffle:
            np.random.shuffle(self.pointers)



    def __len__(self):
        # len of the dataset is the sum of all windows, but we can easily get it from the pointers
        return len(self.pointers)

    def __getitem__(self, idx):
        # get the data and labels using pointer
        _, window = self.pointers[idx]
        if self.feature_columns is None:
            # we drop all the columns that are not embedding. It is "timestep", "path", "video_name" in our data
            # if the error arises anyway, check the columns in your data. Maybe it is better then to explicitly
            # provide the feature_columns?
            embeddings = window.drop(self.label_columns+["timestep", "path", "video_name"], axis=1).values
        else:
            # with the explicitly provided feature_columns, we can easily get the embeddings
            embeddings = window[self.feature_columns].values

        # transform embeddings into tensors
        embeddings = torch.from_numpy(embeddings)
        labels = window[self.label_columns].values
        # preprocess embeddings if needed
        if self.preprocessing_functions is not None:
            embeddings = [self.__preprocess_embeddings(emb) for emb in embeddings]
        # The output shape is (seq_len, num_features)
        # change type to float32
        embeddings = embeddings.type(torch.float32)
        # turn labels into tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        return embeddings, labels


    def __get_key_by_idx__(self, idx):
        key, window = self.pointers[idx]
        return key

    def get_sequence_length(self):
        """ Returns the length of the sequence. """
        return self.pointers[0][1].shape[0]


    def __preprocess_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        for func in self.preprocessing_functions:
            embeddings = func(embeddings)
        return embeddings


    def __cut_all_data_on_windows(self):
        """ Cuts all data on windows. """
        self.cut_windows = OrderedDict()
        for key, frames in self.embeddings_with_labels.items():
            if self.only_consecutive_windows:
                self.cut_windows[key] = self.__cut_sequence_on_consecutive_windows(frames, self.window_size, self.stride)
            else:
                self.cut_windows[key] = self.__create_windows_out_of_frames(frames, self.window_size, self.stride)
        # check if there were some sequences with not enough frames to create a window
        # they have been returned as None, so we need to remove them
        self.cut_windows = OrderedDict({key: windows for key, windows in self.cut_windows.items() if windows is not None})



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
            timestep = min(frames['timestep'].iloc[1]-frames['timestep'].iloc[0], frames['timestep'].iloc[-1]-frames['timestep'].iloc[-2])
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


    def __cut_sequence_on_consecutive_windows(self, sequence:pd.DataFrame, window_size:int, stride:int)->Union[List[pd.DataFrame],None]:
        # check if the sequence is long enough
        # if not, return None and this sequence will be skipped in the __cut_all_data_on_windows method
        if sequence.shape[0] < window_size:
            return None
        # calculate the number of frames in the window
        if self.consider_timestamps:
            timestep = min(sequence['timestep'].iloc[1] - sequence['timestep'].iloc[0],
                           sequence['timestep'].iloc[-1] - sequence['timestep'].iloc[-2])
            window_size = int(np.round(window_size / timestep))
            # stride in seconds needs to be converted to number of frames
            stride = int(np.round(stride / timestep))
        # the problem is that the timesteps are not monotonically increasing. THerefore, we need to cut the data on windows
        # so that the timesteps within the window increase monotonically (with the same timestep/value).
        # The procedure is the following:
        # 1. Get the value of first timestep
        # 2. Get the value of last timestep
        # 3. Get the difference between the first and the last timestep
        # 4. Calculate the value that should be between the first and the last timestep depending on the size of the window
        # 5. Compare the value from the step 4 with the value from the step 3. If it is not equal, do not take this window
        # 6. If it is equal, take the window
        # !!! IMPORTANT!!! we cut windows in this function based on frame_num column, not on the timestep column
        windows = []
        # cut sequence on windows using while and shifting the window every step
        window_start = 0
        window_end = window_start + window_size
        while window_end <= len(sequence):
            window = sequence.iloc[window_start:window_end]
            # check if the timesteps are monotonically increasing
            first_timestep = window['frame_num'].iloc[0]
            last_timestep = window['frame_num'].iloc[-1]
            # take the most often difference between the timesteps
            timestep_difference = window['frame_num'].diff().round(2).mode().values[0]
            # calculate actual range in timesteps and the value that should be in case we have monotonically increasing timesteps
            actual_range = np.round(last_timestep - first_timestep, 2)
            reference_range = np.round(timestep_difference * (window_size-1), 2)
            if actual_range == reference_range:
                windows.append(window)
            window_start += stride
            window_end += stride
        # also add the last window as it is usually ignored
        window_start = len(sequence)-window_size
        window_end = len(sequence)
        start_timestamp = sequence['frame_num'].iloc[window_start]
        end_timestamp = sequence['frame_num'].iloc[window_end-1]
        timestep_difference = end_timestamp - start_timestamp
        timestep_value = start_timestamp + (window_size-1)*timestep_difference
        if timestep_value == end_timestamp:
            windows.append(sequence.iloc[-window_size:])
        return windows


