import gc
from typing import Dict, Iterable, Tuple, Callable

import tensorflow as tf
import numpy as np
import wandb


class WandB_LR_log_callback(tf.keras.callbacks.Callback):

    def __init__(self):
        super(WandB_LR_log_callback, self).__init__()

    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.learning_rate.numpy().flatten()[0]
        wandb.log({"lr": lr}, commit=False)


class WandB_val_metrics_callback(tf.keras.callbacks.Callback):

    def __init__(self, data_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]]):
        super(WandB_val_metrics_callback, self).__init__()
        self.data_generator = data_generator
        self.metrics = metrics

    def calculate_metrics(self) -> Dict[str, float]:
        # make predictions for data from generator and save ground truth labels
        total_predictions = np.zeros((0,))
        total_ground_truth = np.zeros((0,))
        for x, y in self.data_generator:
            predictions = self.model.predict(x, batch_size=16)
            predictions = predictions.argmax(axis=-1).reshape((-1,))
            total_predictions = np.append(total_predictions, predictions)
            total_ground_truth = np.append(total_ground_truth, y.argmax(axis=-1).reshape((-1,)))
        # calculate all provided metrics and save them as dict object
        # as Dict[metric_name->value]
        results = {}
        for key in self.metrics.keys():
            results[key] = self.metrics[key](total_ground_truth, total_predictions)
        # clear RAM
        del total_predictions, total_ground_truth
        gc.collect()
        return results

    def on_epoch_end(self, epoch, logs):
        # calculate all metrics using provided data_generator
        metric_values = self.calculate_metrics()
        # log them
        wandb.log(metric_values, commit=False)
