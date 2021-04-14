#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
from typing import Tuple

from PIL import Image
import numpy as np
import pandas as pd

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


def load_image(path:str)-> np.ndarray:
    # TODO: write description
    with Image.open(path) as im:
        return np.ndarray(im)

def save_image(img:np.ndarray, path_to_output:str)->None:
    # TODO: write description
    img = Image.fromarray(img)
    img.save(path_to_output)

def resize_image(img:np.ndarray, new_size:Tuple[int, int])-> np.ndarray:
    # TODO: write description
    img=Image.fromarray(img)
    img=img.resize(new_size)
    return np.array(img)