#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""

from typing import Optional

from tensorflow.keras.utils import Sequence
import pandas as pd




class ImageDataLoader(Sequence):
    """TODO:write description"""
    horizontal_flip:bool
    vertical_flip:bool
    horizontal_shift:float
    vertical_shift:float
    brightness:float
    shearing:float
    zooming:float
    rotation:int
    scaling:float
    channel_random_noise:float
    mixup:bool
    paths_with_labels:pd.DataFrame

    def __init__(self, paths_with_labels:pd.DataFrame,
                 horizontal_flip:Optional[bool]=None, vertical_flip:Optional[bool]=None, horizontal_shift:Optional[float]=None,
                 vertical_shift:Optional[float]=None, brightness:Optional[float]=None, shearing:Optional[float]=None,
                 zooming:Optional[float]=None, rotation:Optional[int]=None, scaling:Optional[float]=None,
                 channel_random_noise:Optional[float]=None, mixup:Optional[bool]=None):
        # TODO: write description
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.horizontal_shift=horizontal_shift
        self.vertical_shift=vertical_shift
        self.brightness=brightness
        self.shearing=shearing
        self.zooming=zooming
        self.rotation=rotation
        self.scaling=scaling
        self.channel_random_noise=channel_random_noise
        self.mixup=mixup
        self.paths_with_labels=paths_with_labels
        # checking the provided DataFrame
        if paths_with_labels.columns!=['filename', 'class']:
            raise AttributeError('DataFrame columns should be \'filename\' and \'class\'. Got %s.'%paths_with_labels.columns)
        if paths_with_labels.shape[0]==0:
            raise AttributeError('DataFrame is empty.')






    def on_epoch_end(self):
        # TODO: write description
        pass

    def __getitem__(self, index):
        # TODO: write description
        pass

    def __len__(self)-> int:
        # TODO: write description
        pass
