import gc
import os
from typing import Dict, Iterable, Tuple, Callable, Optional

import tensorflow as tf
import numpy as np
import wandb


class WandB_LR_log_callback(tf.keras.callbacks.Callback):

    def __init__(self):
        super(WandB_LR_log_callback, self).__init__()

    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.learning_rate.numpy().flatten()[0]
        wandb.log({"lr": lr}, commit=False)


class WandB_val_metrics_callback_depricated(tf.keras.callbacks.Callback):

    def __init__(self, data_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
                 metric_to_monitor:Optional[str]=None):
        super(WandB_val_metrics_callback_depricated, self).__init__()
        self.data_generator = data_generator
        self.metrics = metrics
        self.metric_to_monitor = metric_to_monitor
        self.best_metric_value=0


    def calculate_metrics(self) -> Dict[str, float]:
        # make predictions for data from generator and save ground truth labels
        total_predictions = np.zeros((0,))
        total_ground_truth = np.zeros((0,))
        for x, y in self.data_generator:
            predictions = self.model.predict(x, batch_size=64)
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
        print('val_metrics:', metric_values)
        # log them
        wandb.log(metric_values, commit=False)
        # save best model based on defined metric if needed
        if self.metric_to_monitor:
            if self.best_metric_value<=metric_values[self.metric_to_monitor]:
                self.best_metric_value = metric_values[self.metric_to_monitor]
                self.model.save_weights(os.path.join(wandb.run.dir, "model_best_%s.h5"%self.metric_to_monitor))
        # clear multiprocessing Pool RAM if needed
        if self.data_generator.pool is not None:
            self.data_generator._realise_multiprocessing_pool()
            print('validation pool cleaned')




class WandB_val_metrics_callback(tf.keras.callbacks.Callback):

    def __init__(self, data_generator: tf.data.Dataset,
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
                 metric_to_monitor:Optional[str]=None,
                 log_class_distribution:bool=False):
        super(WandB_val_metrics_callback, self).__init__()
        self.data_generator = data_generator
        self.metrics = metrics
        self.metric_to_monitor = metric_to_monitor
        self.best_metric_value=0
        self.log_class_distribution = log_class_distribution


    def calculate_metrics(self) -> Dict[str, float]:
        # make predictions for data from generator and save ground truth labels
        total_predictions = np.zeros((0,))
        total_ground_truth = np.zeros((0,))
        if self.log_class_distribution:
            total_class_distribution = None
        for x, y in self.data_generator.as_numpy_iterator():
            predictions = self.model.predict(x, batch_size=64).squeeze()
            # log class distribution if needed
            if self.log_class_distribution:
                total_class_distribution = predictions.sum(axis=0) if total_class_distribution is None \
                    else total_class_distribution+predictions.sum(axis=0)
            # proceed with predictions and ground truth labels
            predictions = predictions.argmax(axis=-1).reshape((-1,))
            total_predictions = np.append(total_predictions, predictions)
            total_ground_truth = np.append(total_ground_truth, y.squeeze().argmax(axis=-1).reshape((-1,)))
        # normalize class distribution if needed
        if self.log_class_distribution:
            total_class_distribution = total_class_distribution/total_class_distribution.sum()
            total_class_distribution = total_class_distribution.reshape((-1,))
            wandb.log({"class_%i_distribution"%i:total_class_distribution[i] for i in range(total_class_distribution.shape[0])},
                      commit=False)
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
        print('val_metrics:', metric_values)
        # log them
        wandb.log(metric_values, commit=False)
        # save best model based on defined metric if needed
        if self.metric_to_monitor:
            if self.best_metric_value<=metric_values[self.metric_to_monitor]:
                self.best_metric_value = metric_values[self.metric_to_monitor]
                self.model.save_weights(os.path.join(wandb.run.dir, "model_best_%s.h5"%self.metric_to_monitor))


    def on_train_end(self, logs):
        wandb.config.update({'best_%s' % (self.metric_to_monitor):
                                 self.best_metric_value})
        print("------------------------------BEST VALUE WAS SAVED-------------------------")

