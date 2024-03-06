import os
from functools import partial
from typing import Callable, List, Dict, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataLoader(Dataset):
    def __init__(self, paths_with_labels:pd.DataFrame, preprocessing_functions:List[Callable]=None,
                 augmentation_functions:Dict[Callable, float]=None, shuffle:bool=False,
                 output_labels:Optional[bool]=True, labels_columns:Optional[List[str]]=None,
                 output_paths:Optional[bool]=False):
        """Image data loader for PyTorch models. Apart from the loading on-the-fly, it preprocesses and augments images if specified.
           paths_with_labels should be passed as a pandas DataFrame with following columns: ['path','label_0','label_1',...,'label_n'].

        :param paths_with_labels: pd.DataFrame
                Pandas DataFrame with paths to images and labels. Columns format: ['path','label_0','label_1',...,'label_n']
        :param preprocessing_functions: List[Callable]
                List of preprocessing functions, each function should take and output one image. Preferably torch.Tensor.
        :param augmentation_functions: Dict[Callable, float]
                Dictionary of augmentation functions and probabilities of applying them. Each function should take and output one image. Preferably torch.Tensor.
        :param shuffle: bool
                Whether to shuffle data or not.
        :param output_labels: Optional[bool]
                Whether to output labels or not. If True, then __getitem__ will return image and label, else only image.
        :param labels_columns: Optional[List[str]]
                List of columns with labels. If None, then all columns except 'path' will be considered as labels.
        :param output_paths: Optional[bool]
                Whether to output paths or not. If True, then __getitem__ will return path in addition to the overall output.
        """
        self.paths_with_labels = paths_with_labels
        # shuffle data if specified
        if shuffle:
            self.paths_with_labels = self.paths_with_labels.sample(frac=1).reset_index(drop=True)
        self.output_labels = output_labels
        self.output_paths = output_paths
        # divide paths_with_labels into paths and labels
        self.img_paths = self.paths_with_labels[['path']]
        if self.output_labels:
            if labels_columns:
                self.labels = self.paths_with_labels[labels_columns]
            else:
                self.labels = self.paths_with_labels.drop(columns=['path'])
        del self.paths_with_labels
        self.preprocessing_functions = preprocessing_functions
        self.augmentation_functions = augmentation_functions

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        image = read_image(self.img_paths.iloc[idx, 0])
        # turn grey image into RGB if needed
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # apply augmentation if needed
        if self.augmentation_functions:
            image = self.augmentation(image)
        # apply preprocessing
        image = self.preprocess_image(image)
        output = [image]
        if self.output_labels:
            label = self.labels.iloc[idx].values.astype(np.float32)
            output = output + [label]
        if self.output_paths:
            output = output + [self.img_paths.iloc[idx, 0]]
        return tuple(output)

    def preprocess_image(self, image:torch.Tensor)->torch.Tensor:
        for func in self.preprocessing_functions:
            image = func(image)
        return image

    def augmentation(self, image:torch.Tensor)->torch.Tensor:
        for func, prob in self.augmentation_functions.items():
            if torch.rand(1) < prob:
                image = func(image)
        return image

    def get_num_classes(self)->int:
        return self.labels.shape[1]