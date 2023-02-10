import os
from functools import partial
from typing import Callable, List, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataLoader(Dataset):
    def __init__(self, paths_with_labels:pd.DataFrame, preprocessing_functions:List[Callable]=None,
                 augmentation_functions:Dict[Callable, float]=None, shuffle:bool=True):
        """Image data loader for PyTorch models. Apart from the loading on-the-fly, it preprocesses and augments images if specified.
           paths_with_labels should be passed as a pandas DataFrame with following columns: ['path','label_0','label_1',...,'label_n'].

        :param paths_with_labels: pd.DataFrame
                Pandas DataFrame with paths to images and labels. Columns format: ['path','label_0','label_1',...,'label_n']
        :param preprocessing_functions: List[Callable]
                List of preprocessing functions, each function should take and output one image. Preferably torch.Tensor.
        :param augmentation_functions: Dict[Callable, float]
                Dictionary of augmentation functions and probabilities of applying them. Each function should take and output one image. Preferably torch.Tensor.
        """
        self.paths_with_labels = paths_with_labels
        # shuffle data if specified
        if shuffle:
            self.paths_with_labels = self.paths_with_labels.sample(frac=1).reset_index(drop=True)
        # divide paths_with_labels into paths and labels
        self.img_paths = self.paths_with_labels[['path']]
        self.labels = self.paths_with_labels.drop(columns=['path'])
        del self.paths_with_labels
        self.preprocessing_functions = preprocessing_functions
        self.augmentation_functions = augmentation_functions

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = read_image(self.img_paths.iloc[idx, 0])
        if self.augmentation_functions:
            image = self.augmentation(image)
        image = self.preprocess_image(image)
        label = self.labels.iloc[idx].values.astype(np.float32)
        return image, label

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