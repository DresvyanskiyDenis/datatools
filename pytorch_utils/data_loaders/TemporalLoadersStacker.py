from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader


class TemporalEmbeddingsLoaders_multi(Dataset):

    def __init__(self, embeddings_with_labels_list:List[Dict[str, pd.DataFrame]], label_columns:List[str],
                 window_size:Union[int, float], stride:Union[int, float],
                 consider_timestamps:Optional[bool]=False,
                 preprocessing_functions:List[Callable]=None, shuffle:bool=False):
        super().__init__()
        self.embeddings_with_labels_list = embeddings_with_labels_list
        self.label_columns = label_columns
        self.window_size = window_size
        self.stride = stride
        self.consider_timestamps = consider_timestamps
        self.preprocessing_functions = preprocessing_functions
        self.shuffle = shuffle

        # reorder other embeddings with labels according to the order of the first one. Order will be ensured by
        # using OrderedDict
        first, others = self.__reorder_other_dicts_according_first_one(embeddings_with_labels_list[0],
                                                                       embeddings_with_labels_list[1:])
        self.embeddings_with_labels_list = [first] + others
        # cut all data on windows
        self.




    def __reorder_other_dicts_according_first_one(self, first:Dict[str, pd.DataFrame],
                                                  other:List[Dict[str, pd.DataFrame]]) -> \
            Tuple[Dict[str, pd.DataFrame],
                  List[Dict[str, pd.DataFrame]]
            ]:
        """ Reorders other dicts according to the order of the first one. """
        # make order for the first dict
        first_reordered = OrderedDict()
        for key in first.keys():
            first_reordered[key] = first[key]
        # make order for other dicts as it is in the first one
        other_reordered = []
        for other_dict in other:
            reordered = OrderedDict()
            for key in first.keys():
                reordered[key] = other_dict[key]
            other_reordered.append(reordered)
        return first_reordered, other_reordered


    def __len__(self):
        # len of the dataset is the sum of all windows, but we can easily get it from the pointers
        return len(self.pointers)

    def __getitem__(self, idx):
        # get the data and labels using pointer
        _, window = self.pointers[idx]
        embeddings = window.drop(columns=self.label_columns+['timestep', 'video_name', 'path']).values
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

    def get_sequence_length(self):
        """ Returns the length of the sequence. """
        return self.pointers[0][1].shape[0]


    def __preprocess_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        for func in self.preprocessing_functions:
            embeddings = func(embeddings)
        return embeddings


    def __cut_all_data_on_windows(self, data:Dict[str, pd.DataFrame])->Dict[str, List[pd.DataFrame]]:
        """ Cuts all data on windows. """
        cut_windows = {}
        for key, frames in data.items():
            cut_windows[key] = self.__create_windows_out_of_frames(frames, self.window_size, self.stride)
        # check if there were some sequences with not enough frames to create a window
        # they have been returned as None, so we need to remove them
        cut_windows = {key: windows for key, windows in self.cut_windows.items() if windows is not None}
        return cut_windows



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


