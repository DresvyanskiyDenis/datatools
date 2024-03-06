#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module for extracting embeddings from images using torch models.

    List of functions:
        * EmbeddingsExtractor - class for extracting embeddings from images using torch models.
        Only for Frame-wise extraction/models
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2023"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import gc
from typing import Optional, Callable, List, Union
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorch_utils.data_loaders.ImageDataLoader_new import ImageDataLoader

data_types = Union[np.ndarray,
                   torch.Tensor,
                   List[Union[np.ndarray, torch.Tensor]],
                   pd.DataFrame]


class EmbeddingsExtractor():
    """ Extracts embeddings using provided torch model. It can extract embeddings from:
    - One image represented as numpy array or torch tensor
    - List of images
    - Dataframe with paths to images

    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None,
                 preprocessing_functions: List[Callable] = None, output_shape: Optional[int] = None):
        """

        :param model: torch.nn.Module
                The model to extract embeddings from.
        :param preprocessing_functions: List[Callable]
                List of preprocessing functions to apply to the image before feeding it to the model.
                If None, then no preprocessing will be applied.
        :param output_shape: int
                The shape of the output of the model. If None, then it will be inferred from the model.
        :param device: device to use. If None, then device will be equal to the torch.device('cuda') if available, else torch.device('cpu')
        """
        self.model = model
        self.device = device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # host model on device
        self.model.to(self.device)
        self.preprocessing_functions = preprocessing_functions
        # get the output shape of the model
        if output_shape is None:
            self.output_shape = self.__get_output_shape_model()
        else:
            self.output_shape = output_shape


    def __get_output_shape_model(self)->int:
        tmp_idx = -1000
        for i in range(len(self.model) - 1, -1, -1):
            if hasattr(self.model[i], 'out_features'):
                tmp_idx = i
                break
        if tmp_idx == -1000:
            raise ValueError("Cannot find the output shape of the model. Probably, it has no linear layer.")
        return self.model[tmp_idx].out_features

    def extract_embeddings(self, data: data_types, *, batch_size: Optional[int] = None,
                           num_workers: Optional[int] = None, labels_columns: Optional[List[str]] = None,
                           output_path: Optional[str] = None, verbose: Optional[bool] = False) \
            -> Union[torch.Tensor, None]:
        """ Extracts embeddings from provided data. Data can be represented as:
            - One image represented as numpy array or torch tensor
            - List of images
            - Dataframe with paths to images

        :param data: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]], pd.DataFrame]
                The data to extract embeddings from.
        :param batch_size: Optional[int]
                The batch size to use. If None, then batch_size will be equal to the 1. Used only if data is
                pd.DataFrame or torch.utils.data.DataLoader.
        :param num_workers: Optional[int]
                The number of workers to use. If None, then num_workers will be equal to the 1. Used only if data is
                pd.DataFrame or torch.utils.data.DataLoader.
        :param output_path: Optional[str]
                The path to save the embeddings. If None, then embeddings will not be saved and returned by this function.
        :param verbose: Optional[bool]
                If True, then prints the progress of extraction.
        :return: Union[np.ndarray, torch.Tensor, None]
        """
        # check the data type
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            embeddings = self.__extract_embeddings_one_image(data)
        elif isinstance(data, list):
            embeddings = self.__extract_embeddings_list_images(data)
        elif isinstance(data, pd.DataFrame):
            embeddings = self.__extract_embeddings_dataframe(data, batch_size=batch_size, num_workers=num_workers,
                                                             verbose=verbose, output_path=output_path,
                                                             labels_columns=labels_columns)
        else:
            raise TypeError(f"Unknown data type {type(data)}")
        if output_path is not None:
            return None
        return embeddings.detach().cpu().numpy()

    def __extract_embeddings_one_image(self, image: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor]:
        """ Extracts embeddings from one image represented as numpy array or torch tensor.

        :param image: Union[np.ndarray, torch.Tensor]
                The image to extract embeddings from.
        :return: Union[np.ndarray, torch.Tensor]
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        # preprocessing
        if self.preprocessing_functions is not None:
            for func in self.preprocessing_functions:
                image = func(image)
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        with torch.no_grad():
            embeddings = self.model(image)
        return embeddings

    def __extract_embeddings_list_images(self, images: List[Union[np.ndarray, torch.Tensor]]) -> \
            Union[torch.Tensor]:
        """ Extracts embeddings from list of images represented as numpy array or torch tensor.

        :param images: List[Union[np.ndarray, torch.Tensor]]
                The list of images to extract embeddings from.
        :return: Union[np.ndarray, torch.Tensor]
        """
        embeddings = []
        for image in images:
            embeddings.append(self.__extract_embeddings_one_image(image))
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def __extract_embeddings_dataframe(self, dataframe: pd.DataFrame, *, batch_size: Optional[int] = None,
                                       labels_columns: Optional[List[str]] = None,
                                       num_workers: Optional[int] = None, verbose: Optional[bool] = False,
                                       output_path: str) -> None:
        """ Extracts embeddings from dataframe with paths to images.

        :param dataframe: pd.DataFrame
             The dataframe with paths to images.
        :param batch_size: Optional[int]
             The batch size to use. If None, then batch_size will be equal to the 1.
        :param num_workers: Optional[int]
             The number of workers to use. If None, then num_workers will be equal to the 1.
        :param verbose: Optional[bool]
             If True, then prints the progress of extraction.
        :param output_path: Optional[str]
                The path to save the embeddings. This should be the path to the csv file.
        :return: None
        """
        if batch_size is None:
            batch_size = 1
        if num_workers is None:
            num_workers = 1
        # to do so, we need to check the columns names of the dataframe. The most important is that the first column
        # should be the column with paths to images and have the name 'path'
        if dataframe.columns[0] != 'path':
            raise ValueError("The first column of the dataframe should be the column with paths to images and "
                             "have the name 'path'")
        # create dataloader from dataframe
        dataloader = ImageDataLoader(paths_with_labels=dataframe, preprocessing_functions=self.preprocessing_functions,
                                     augmentation_functions=None, shuffle=False, output_paths=True,
                                     output_labels=False if labels_columns is None else True,
                                     labels_columns=labels_columns)
        dataloader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size,
                                                 num_workers=num_workers, shuffle=False)
        # extract embeddings. Remember that to alleviate the usage of those in the future, we will save the paths to images
        # and the embeddings in the csv file, the first column of which will be the "path" and all consequent columns
        # will be the values of embeddings with names "embedding_0", "embedding_1", etc.
        # Also, to save the memory consumption, we will save the embeddings in the csv file batch-wise
        self.__extract_embeddings_from_dataloader_and_save_them(dataloader=dataloader, output_path=output_path,
                                                                verbose=verbose, labels_columns=labels_columns)

    def __extract_embeddings_from_dataloader_and_save_them(self, dataloader: torch.utils.data.DataLoader,
                                                           output_path: str = None,
                                                           labels_columns: Optional[List[str]] = None,
                                                           verbose: Optional[bool] = False) -> None:
        """ Extracts embeddings from dataloader and saves them to the csv file.

        :param dataloader: torch.utils.data.DataLoader
                The dataloader to extract embeddings from.
        :param output_path: Optional[str]
                The path to save the embeddings. This should be the path to the csv file.
        :param labels_columns: Optional[List[str]]
                List of columns with labels. If None, then all columns except 'path' will be considered as labels.
        :param verbose: Optional[bool]
                If True, then prints the progress of extraction.
        :return: None
        """
        # To save the memory consumption, we will save the embeddings in the csv file batch-wise
        # create the list of columns names
        columns_names = ['path']
        for i in range(self.output_shape):
            columns_names.append('embedding_' + str(i))
        if labels_columns is not None:
            columns_names = columns_names + labels_columns
        # extract embeddings batch-wise and save them first to dataframe, then dump to the csv file, and then clear
        # the dataframe
        # create empty csv file to dump the columns names
        pd.DataFrame(columns=columns_names).to_csv(output_path, index=False)
        for data in tqdm(dataloader, desc='Extracting embeddings', disable=not verbose):
            if labels_columns is not None:
                batch, labels, paths = data
            else:
                batch, paths = data
            batch = batch.float().to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(batch)
            batch_embeddings = batch_embeddings.cpu().numpy()
            if labels_columns is not None:
                labels = labels.cpu().numpy()
                batch_embeddings = np.concatenate([batch_embeddings, labels], axis=1)
            paths = np.array(paths)
            # concatenate one column of zeros to the batch_embeddings to replace it later with the paths to images
            batch_embeddings = np.concatenate([np.zeros((paths.shape[0], 1)), batch_embeddings], axis=1)
            # create dataframe
            batch_embeddings = pd.DataFrame(batch_embeddings, columns=columns_names)
            # replace the first column with the paths to images
            batch_embeddings['path'] = paths
            # dump to the csv file, writing on top of the existing data
            batch_embeddings.to_csv(output_path, index=False, mode='a', header=None)

