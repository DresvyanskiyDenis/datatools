#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides functions for class weighting. Works only for categorical/integer classes.

List of functions:

    * get_class_weights_Effective_Number_of_Samples - gives class weights by using Effective Number Of Samples (https://arxiv.org/pdf/1901.05555.pdf)

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


from typing import Dict

import numpy as np


def get_class_weights_Effective_Number_of_Samples(labels:np.ndarray, beta:float)->Dict[int, float]:
    """Calculates the class weights by method "Effective Number of Samples".
    link: https://arxiv.org/pdf/1901.05555.pdf

    :param labels: np.ndarray
            1D numpy array with integer labels
    :param beta: float
            parameter of the Effective Number of Samples
    :return: Dict[int, float]
            Class weights in Dict[class_num->class_weight]
    """
    # https://arxiv.org/pdf/1901.05555.pdf
    if len(labels.shape)!=1:
        raise AttributeError('Passed labels should be 1-dimensional.')
    if beta<0 or beta>=1:
        raise AttributeError('The value of oassed parameter beta should be between 0 and 1 (excluding 1).')
    # calculate the number of samples in each class
    counts=np.unique(labels, return_counts=True)[1].reshape((-1,))
    # transform veta to array for broadcasting
    beta=np.ones((counts.shape[0],))*beta
    # calculate effective num
    effective_num=(1.0-np.power(beta, counts))/(1.0-beta)
    # calculate weights
    weights=1./effective_num
    # normalize it to sum of weights = 1
    weights=weights/np.sum(weights)
    # convert it to dict
    weights=dict((i, weights[i]) for i in range(weights.shape[0]))
    return weights

