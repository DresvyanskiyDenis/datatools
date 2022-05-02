from typing import Tuple, Optional, List, Callable

import pandas as pd
import tensorflow as tf

Tensorflow_Callable = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def get_tensorflow_sequence_loader(paths_and_labels: pd.DataFrame, batch_size: int, augmentation: bool = False,
                             augmentation_methods: Optional[List[Tensorflow_Callable]] = None,
                             preprocessing_function: Optional[Tensorflow_Callable] = None,
                             clip_values: Optional[bool] = None,
                             cache_loaded_images:Optional[bool]=None) -> tf.data.Dataset:
    pass


