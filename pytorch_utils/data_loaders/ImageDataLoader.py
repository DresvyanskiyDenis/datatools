import os
from functools import partial
from typing import Callable, List, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

from pytorch_utils.data_loaders.pytorch_augmentations import pad_image_random_factor, grayscale_image, \
    collor_jitter_image_random, gaussian_blur_image_random, random_perspective_image, random_rotation_image, \
    random_crop_image, random_posterize_image, random_adjust_sharpness_image, random_equalize_image, \
    random_horizontal_flip_image, random_vertical_flip_image


class ImageDataLoader(Dataset):
    def __init__(self, labels:pd.DataFrame, paths_to_images:pd.DataFrame, paths_prefix:str=None, preprocessing_functions:List[Callable]=None,
                 augment:bool=False, augmentation_functions:Dict[Callable, float]=None):
        self.labels = labels
        self.img_paths = paths_to_images

        self.preprocessing_functions = preprocessing_functions
        self.augment = augment
        self.augmentation_functions = augmentation_functions
        self.paths_prefix = '' if paths_prefix is None else paths_prefix

    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, idx):
        image = read_image(os.path.join(self.paths_prefix, self.img_paths.iloc[idx, 0]))
        if self.augment:
            image = self.augmentation(image)
        image = self.preprocess_image(image)
        label = self.labels.iloc[idx].values
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


if __name__ == '__main__':
    paths_to_data=os.listdir(r'C:\Users\Dresvyanskiy\Desktop\013_2016-03-30_Paris\Expert_video')
    paths_to_data = pd.DataFrame(paths_to_data, columns=['path'])
    labels= pd.DataFrame(data=np.array(np.arange(0,len(paths_to_data))), columns=['label'])

    preprocessing_functions = [T.Resize(size=(240,240))]


    augmentation_functions={
        pad_image_random_factor:0.2,
        grayscale_image:0.2,
        partial(collor_jitter_image_random, brightness=0.5, hue=0.3, contrast=0.3, saturation=0.3):0.2,
        partial(gaussian_blur_image_random, kernel_size=(5,9), sigma=(0.1,5)):0.2,
        random_perspective_image:0.2,
        random_rotation_image:0.2,
        partial(random_crop_image, cropping_factor_limits=(0.7,0.9)):0.2,
        random_posterize_image:0.2,
        partial(random_adjust_sharpness_image, sharpness_factor_limits=(0.1, 3)):0.2,
        random_equalize_image:0.2,
        random_horizontal_flip_image:0.2,
        random_vertical_flip_image:0.2
    }

    dataset=ImageDataLoader(labels, paths_to_data, paths_prefix=r'C:\Users\Dresvyanskiy\Desktop\013_2016-03-30_Paris\Expert_video', preprocessing_functions=preprocessing_functions,
                                augment=True, augmentation_functions=augmentation_functions)


    data_loader=torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,
                                            num_workers=4, pin_memory=False)


    for x,y in data_loader:
        print(x.shape, y.shape)