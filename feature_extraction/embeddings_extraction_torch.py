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
                 preprocessing_functions: List[Callable] = None):
        """

        :param model: torch.nn.Module
                The model to extract embeddings from.
        :param device: device to use. If None, then device will be equal to the torch.device('cuda') if available, else torch.device('cpu')
        """
        self.model = model
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # host model on device
        self.model.to(self.device)
        self.preprocessing_functions = preprocessing_functions

    def extract_embeddings(self, data: data_types, *, batch_size: Optional[int] = None,
                           num_workers: Optional[int] = None,
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
                                                             verbose=verbose, output_path=output_path)
        else:
            raise TypeError(f"Unknown data type {type(data)}")
        return embeddings

    def __extract_embeddings_one_image(self, image: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor]:
        """ Extracts embeddings from one image represented as numpy array or torch tensor.

        :param image: Union[np.ndarray, torch.Tensor]
                The image to extract embeddings from.
        :return: Union[np.ndarray, torch.Tensor]
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
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
        :return: Union[np.ndarray, torch.Tensor]
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
                                     augmentation_functions=None, shuffle=False, output_labels=False, output_paths=True)
        dataloader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size,
                                                 num_workers=num_workers, shuffle=False)
        # extract embeddings. Remember that to alleviate the usage of those in the future, we will save the paths to images
        # and the embeddings in the csv file, the first column of which will be the "path" and all consequent columns
        # will be the values of embeddings with names "embedding_0", "embedding_1", etc.
        # Also, to save the memory consumption, we will save the embeddings in the csv file batch-wise
        self.__extract_embeddings_from_dataloader_and_save_them(dataloader=dataloader, output_path=output_path,
                                                                verbose=verbose)

    def __extract_embeddings_from_dataloader_and_save_them(self, dataloader: torch.utils.data.DataLoader,
                                                           output_path: str = None,
                                                           verbose: Optional[bool] = False) -> None:
        """ Extracts embeddings from dataloader and saves them to the csv file.

        :param dataloader: torch.utils.data.DataLoader
                The dataloader to extract embeddings from.
        :param output_path: Optional[str]
                The path to save the embeddings. This should be the path to the csv file.
        :param verbose: Optional[bool]
                If True, then prints the progress of extraction.
        :return: None
        """
        # To save the memory consumption, we will save the embeddings in the csv file batch-wise
        # create the list of columns names
        columns_names = ['path']
        for i in range(self.model.output_dim):
            columns_names.append('embedding_' + str(i))
        # extract embeddings batch-wise and save them first to dataframe, then dump to the csv file, and then clear
        # the dataframe
        dump_counter = 0
        batch_embeddings = None
        for batch, paths in tqdm(dataloader, desc='Extracting embeddings', disable=not verbose):
            batch = batch.to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(batch)
            batch_embeddings = batch_embeddings.cpu().numpy()
            batch_embeddings = np.concatenate([np.expand_dims(paths, axis=1), batch_embeddings], axis=1)
            batch_embeddings = pd.DataFrame(batch_embeddings, columns=columns_names)
            if dump_counter % 100 == 0:
                # dump to the csv file, writing on top of the existing data
                batch_embeddings.to_csv(output_path, index=False, mode='a', header=True)
                batch_embeddings = None
                gc.collect()
            dump_counter += 1
        # dump the rest of the data to the csv file
        if batch_embeddings is not None:
            batch_embeddings.to_csv(output_path, index=False, mode='a', header=True)
            batch_embeddings = None
            gc.collect()
