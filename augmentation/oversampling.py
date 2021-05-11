#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""


from typing import Tuple

import imblearn
import numpy as np
import pandas as pd

def oversample_by_border_SMOTE(data:np.ndarray, labels:np.ndarray, sampling_strategy)->Tuple[np.ndarray, np.ndarray]:
    # TODO: write description
    oversampler=imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=sampling_strategy)
    data, labels=oversampler.fit_resample(data, labels)
    return data, labels