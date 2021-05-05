#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add description
"""


from typing import Tuple

import imblearn
import numpy as np
import pandas as pd

def oversample_by_border_SMOTE(data:np.ndarray, labels:np.ndarray, ratio_of_generating_minor_class:float)->Tuple[np.ndarray, np.ndarray]:
    # TODO: write description
    oversampler=imblearn.over_sampling.BorderlineSMOTE(ratio=ratio_of_generating_minor_class)
    data, labels=oversampler.fit_resample(data, labels)
    return data, labels