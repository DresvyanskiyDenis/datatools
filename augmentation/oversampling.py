#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains oversampling techniques, mostly taken from imbalance learn lib.

imbalanced learn: https://imbalanced-learn.org/stable/

List of functions:

    * oversample_by_border_SMOTE - oversample provided data via borderline SMOTE approach

"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


from typing import Tuple
import imblearn
import numpy as np

def oversample_by_border_SMOTE(data:np.ndarray, labels:np.ndarray, sampling_strategy)->Tuple[np.ndarray, np.ndarray]:
    """Oversample data using borderline SMOTE.

    :param data: np.ndarray
            data to oversample
    :param labels: np.ndarray
            labels related to data (NOT in one-hot encoding format)
    :param sampling_strategy: str
            the strategy of sampling (see imbalanced learn documentation)
    :return: Tuple[np.ndarray, np.ndarray]
            oversamples data and labels
    """
    oversampler=imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=sampling_strategy)
    data, labels=oversampler.fit_resample(data, labels)
    return data, labels