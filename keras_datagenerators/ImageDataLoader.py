#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""

from typing import Optional

from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np

from preprocessing.data_preprocessing.image_preprocessing_utils import shear_image, rotate_image, flip_image, \
    shift_image, change_brightness, zoom_image, crop_image, channel_random_noise, blur_image, \
    get_image_with_worse_quality


class ImageDataLoader(Sequence):
    """TODO:write description"""
    horizontal_flip:float
    vertical_flip:float
    horizontal_shift:float
    vertical_shift:float
    brightness:float
    shearing:float
    zooming:float
    croping:float
    bluring:float
    rotation:float
    scaling:float
    channel_random_noise:float
    worse_quality:float
    mixup:bool

    paths_with_labels:pd.DataFrame
    batch_size:int

    def __init__(self, paths_with_labels:pd.DataFrame,batch_size:int,
                 horizontal_flip:Optional[float]=None, vertical_flip:Optional[float]=None, horizontal_shift:Optional[float]=None,
                 vertical_shift:Optional[float]=None, brightness:Optional[float]=None, shearing:Optional[float]=None,
                 zooming:Optional[float]=None, croping:Optional[float]=None, rotation:Optional[float]=None, scaling:Optional[float]=None,
                 channel_random_noise:Optional[float]=None, bluring:Optional[float]=None, worse_quality:Optional[float]=None,
                 mixup:Optional[bool]=None):
        # TODO: write description
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.horizontal_shift=horizontal_shift
        self.vertical_shift=vertical_shift
        self.brightness=brightness
        self.shearing=shearing
        self.zooming=zooming
        self.croping=croping
        self.rotation=rotation
        self.scaling=scaling
        self.channel_random_noise=channel_random_noise
        self.bluring = bluring
        self.worse_quality=worse_quality
        self.mixup=mixup
        self.paths_with_labels=paths_with_labels
        self.batch_size=batch_size
        # checking the provided DataFrame
        if paths_with_labels.columns!=['filename', 'class']:
            raise AttributeError('DataFrame columns should be \'filename\' and \'class\'. Got %s.'%paths_with_labels.columns)
        if paths_with_labels.shape[0]==0:
            raise AttributeError('DataFrame is empty.')
        # check if all provided variables are in the allowed range (usually, from 0..1 or bool)
        if horizontal_flip<0 or horizontal_flip>1:
            raise AttributeError('Parameter horizontal_flip should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if vertical_flip<0 or vertical_flip>1:
            raise AttributeError('Parameter vertical_flip should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if horizontal_shift<0 or horizontal_shift>1:
            raise AttributeError('Parameter horizontal_shift should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if vertical_shift<0 or vertical_shift>1:
            raise AttributeError('Parameter vertical_shift should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if brightness<0 or brightness>1:
            raise AttributeError('Parameter brightness should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if shearing<0 or shearing>1:
            raise AttributeError('Parameter shearing should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if zooming<0 or zooming>1:
            raise AttributeError('Parameter zooming should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if croping<0 or croping>1:
            raise AttributeError('Parameter croping should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if rotation<0 or rotation>1:
            raise AttributeError('Parameter rotation should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if channel_random_noise<0 or channel_random_noise>1:
            raise AttributeError('Parameter channel_random_noise should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if bluring<0 or bluring>1:
            raise AttributeError('Parameter bluring should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if worse_quality<0 or worse_quality>1:
            raise AttributeError('Parameter worse_quality should be float number between 0 and 1, '
                                 'representing the probability of arising such augmentation technique.')
        if not isinstance(mixup, bool):
            raise AttributeError('Parameter mixup should have bool type.')

    def _shear_image(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[-0.5, 0.5]
        randomly_picked_param=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        img=shear_image(img, shear_factor=randomly_picked_param)
        return img

    def _rotate_image(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[-90, 90]
        randomly_picked_param=np.random.randint(parameter_boundaries[0], parameter_boundaries[1]+1)
        img=rotate_image(img, rotation_angle=randomly_picked_param)
        return img

    def _flip_image_vertical(self, img:np.ndarray):
        # TODO: write description
        img=flip_image(img, flip_type='vertical')
        return img

    def _flip_image_horizontal(self, img:np.ndarray):
        # TODO: write description
        img=flip_image(img, flip_type='horizontal')
        return img

    def _shift_image(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[-50, 50]
        randomly_picked_param_vertical=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        randomly_picked_param_horizontal = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        img=shift_image(img, shift_vector=(randomly_picked_param_horizontal, randomly_picked_param_vertical))
        return img

    def _change_brightness_image(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[-0.5, 0.5]
        randomly_picked_param=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        img=change_brightness(img, brightness_factor=randomly_picked_param)
        return img

    def _zoom_image(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[1, 3]
        old_shape=img.shape[:2]
        randomly_picked_param=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        # firstly zoom image (increase or decrease the size)
        img=zoom_image(img, zoom_factor=randomly_picked_param)
        # take part of image (for example, with size of old shape, if it became bigger)
        # x0 can be chosen in the range (0, current_img.shape[1]-old_img.shape[1])
        # y0 can be chosen in the range (0, current_img.shape[0]-old_img.shape[0])
        x0 = np.random.randint(0, img.shape[1]-old_shape[1])
        y0 = np.random.randint(0, img.shape[0] - old_shape[0])
        x1=x0+old_shape[1]
        y1=y0+old_shape[1]
        img=crop_image(img, bbox=(x0,y0,x1,y1))
        return img


    def _add_noise_on_one_channel(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[40, 100]
        randomly_picked_param=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        randomly_picked_channel=np.random.randint(0,3)
        img=channel_random_noise(img, num_channel=randomly_picked_channel, std=randomly_picked_param)
        return img

    def _random_cutting_out(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        # choose which proportion of image should we cut out
        parameter_boundaries=[0.2, 0.4]
        randomly_picked_param=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        # transform it to index according to shape f image
        randomly_picked_param=int(np.round(randomly_picked_param*(img.shape[0]+img.shape[1])/2.))
        # calculate randomly the area, which will be cut out
        x0=np.random.randint(0, img.shape[1]-randomly_picked_param)
        x1=x0+randomly_picked_param
        y0 = np.random.randint(0, img.shape[0] - randomly_picked_param)
        y1=y0+randomly_picked_param
        # copy for getting rid of non-predictable changing the image
        img=img.copy()
        # cut out
        img[y0:y1, x0:x1]=0
        return img

    def _blur_image(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[1,3]
        randomly_picked_param=np.random.randint(parameter_boundaries[0], parameter_boundaries[1]+1)
        img=blur_image(img, sigma=randomly_picked_param)
        return img

    def _worse_quality(self, img:np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries=[0.2, 0.8]
        randomly_picked_param=np.random.uniform(parameter_boundaries[0], parameter_boundaries[1]+1)
        img=get_image_with_worse_quality(img, rescaling_factor=randomly_picked_param)
        return img


    def _preprocess_one_image(self, img:np.ndarray):
        # TODO: implement it
        pass


    def on_epoch_end(self):
        # TODO: write description
        # just shuffle rows in dataframe
        self.paths_with_labels=self.paths_with_labels.sample(frac=1)

    def __getitem__(self, index):
        # TODO: write description
        # TODO: implement it
        pass

    def __len__(self)-> int:
        # TODO: write description
        num_steps=np.ceil(self.paths_with_labels.shape[0]/self.batch_size)
        return num_steps
