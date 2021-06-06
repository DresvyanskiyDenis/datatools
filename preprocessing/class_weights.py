#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    # TODO: add description
"""
from typing import Dict, Union

import numpy as np
import pandas as pd


def get_class_weights_Effective_Number_of_Samples(labels:np.ndarray, beta:float)->Dict[int, float]:
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




if __name__=="__main__":
    labels=np.array([3, 1, 3, 1, 1, 1, 1, 1, 0, 0, 0, 2])
    weights=get_class_weights_Effective_Number_of_Samples(labels, 0.99)
    print(weights)