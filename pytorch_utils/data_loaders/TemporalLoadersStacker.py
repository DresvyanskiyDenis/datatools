from collections import OrderedDict
from typing import Callable, List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader


class TemporalLoadersStacker(Dataset):

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
        self.embeddings_with_labels_list = self.__reorder_other_dicts_according_first_one(embeddings_with_labels_list[0],
                                                                       embeddings_with_labels_list[1:])
        # create loaders for every dict
        self.loaders = []
        for embeddings_with_labels in self.embeddings_with_labels_list:
            loader = TemporalEmbeddingsLoader(embeddings_with_labels=embeddings_with_labels,
                                              label_columns=label_columns,
                                              feature_columns=None,
                                              window_size=window_size,
                                              stride=stride,
                                              consider_timestamps=consider_timestamps,
                                              preprocessing_functions=preprocessing_functions,
                                              shuffle=False)
            self.loaders.append(loader)

        # check congruety of the loaders
        self.__check_congruety_of_loaders()
        print("Congruety of the loaders is OK.")



    def __check_congruety_of_loaders(self):
        # check congruety of the loaders
        for loader in self.loaders:
            if loader.__len__() != self.loaders[0].__len__():
                raise ValueError("Length of the loaders should be the same.")
        # check congruety according to keys. The reference is the first loader
        for idx in range(self.loaders[0].__len__()):
            key = self.loaders[0].__get_key_by_idx__(idx)
            data, labels = self.loaders[0].__getitem__(idx)
            for loader in self.loaders[1:]:
                if key != loader.__get_key_by_idx__(idx):
                    raise ValueError("Keys of the loaders should be the same.")
            # check the labels if they are the same
            """for loader in self.loaders[1:]:
                _, labels_ = loader.__getitem__(idx)
                if not np.array_equal(labels, labels_):
                    raise ValueError("Labels of the loaders should be the same.")"""



    def __reorder_other_dicts_according_first_one(self, first:Dict[str, pd.DataFrame],
                                                  other:List[Dict[str, pd.DataFrame]]) -> List[Dict[str, pd.DataFrame]]:
        """ Reorders other dicts according to the order of the first one. """
        # make order for the first dict
        first_reordered = OrderedDict()
        for key in first.keys():
            first_reordered[key] = first[key]
        # make order for other dicts as it is in the first one
        other_reordered = []
        for other_dict in other:
            reordered = OrderedDict()
            for key in first_reordered.keys():
                reordered[key] = other_dict[key]
            other_reordered.append(reordered)
        # pack into one list all reordered dicts
        result = [first_reordered] + other_reordered
        return result


    def __len__(self):
        # len of the dataset is the sum of all windows, but we can easily get it from the pointers
        return self.loaders[0].__len__()

    def __getitem__(self, idx):
        # get the data and labels using pointers
        # the labels are the same for all loaders, so we can take them from the first loader
        data, labels = self.loaders[0].__getitem__(idx)
        #labels = torch.tensor(labels, dtype=torch.float32)
        data = [data]
        # get data from other loaders
        for loader in self.loaders[1:]:
            data_, _ = loader.__getitem__(idx)
            data.append(data_)
        # return data and labels
        return data, labels

    def get_sequence_length(self):
        """ Returns the length of the sequence. """
        return self.loaders[0].get_sequence_length()



