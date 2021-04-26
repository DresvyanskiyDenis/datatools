from functools import partial
from typing import Callable, Iterable, Tuple, List, Optional, Union, TextIO

import tensorflow as tf
import numpy as np


def get_reduceLRonPlateau_callback(monitoring_loss: str = 'val_loss', reduce_factor: float = 0.1,
                                   num_patient_epochs: int = 10,
                                   min_lr: float = 0.) -> tf.keras.callbacks.ReduceLROnPlateau:
    # TODO: write description
    callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitoring_loss, factor=reduce_factor,
                                                    patience=num_patient_epochs, min_lr=min_lr)
    return callback


def get_LRreduce_callback(scheduller: Callable[[int, float], float]) -> tf.keras.callbacks.LearningRateScheduler:
    # TODO: write description
    callback = tf.keras.callbacks.LearningRateScheduler(scheduller)
    return callback


def get_annealing_LRreduce_callback(highest_lr: float, lowest_lr: float,
                                    annealing_period: int) -> tf.keras.callbacks.LearningRateScheduler:
    # TODO: write description
    learning_rates = np.linspace(start=highest_lr, stop=lowest_lr, num=annealing_period)

    def get_lr_according_to_annealing_step(current_epoch: int, current_lr: float, learning_rates: np.ndarray) -> float:
        annealing_period = learning_rates.shape[0]
        idx_of_annealing_step = current_epoch % annealing_period
        return learning_rates[idx_of_annealing_step]

    annealing_func_for_tf_callback = partial(get_lr_according_to_annealing_step, learning_rates=learning_rates)
    callback = get_LRreduce_callback(annealing_func_for_tf_callback)
    return callback


class best_weights_setter_callback(tf.keras.callbacks.Callback):
    """Calculates the recall score at the end of each training epoch and saves the best weights across all the training
        process. At the end of training process, it will set weights of the model to the best found ones.
        # TODO: write, which types of metric functions it supports
    """

    def __init__(self, val_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                 evaluation_metric: Callable[[np.ndarray, np.ndarray], float]):
        super(best_weights_setter_callback, self).__init__()
        # best_weights to store the weights at which the minimum UAR occurs.
        self.best_weights = None
        # generator to iterate on it on every end of epoch
        self.val_generator = val_generator
        self.evaluation_metric = evaluation_metric

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        # TODO: write description
        current_recall = self.custom_recall_validation_with_generator(self.val_generator, self.model)
        print('current validation %s:' % self.evaluation_metric, current_recall)
        if np.greater(current_recall, self.best):
            self.best = current_recall
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        # TODO: write description
        self.model.set_weights(self.best_weights)

    def custom_recall_validation_with_generator(self, generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                                                model: tf.keras.Model) -> float:
        # TODO: write description
        total_predictions = np.zeros((0,))
        total_ground_truth = np.zeros((0,))
        for x, y in generator:
            predictions = model.predict(x)
            predictions = predictions.argmax(axis=-1).reshape((-1,))
            total_predictions = np.append(total_predictions, predictions)
            total_ground_truth = np.append(total_ground_truth, y.argmax(axis=-1).reshape((-1,)))
        return self.evaluation_metric(total_ground_truth, total_predictions)


class validation_with_generator_callback(tf.keras.callbacks.Callback):
    # TODO: write description

    def __init__(self, val_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                 metrics: Tuple[Callable[[np.ndarray, np.ndarray], float]],
                 logger: Optional[TextIO] = None,
                 num_metric_to_set_weights: Optional[int] = None):
        super(validation_with_generator_callback, self).__init__()
        # best_weights to store the weights at which the minimum UAR occurs.
        self.best_weights = None
        # generator to iterate on it on every end of epoch
        self.val_generator = val_generator
        self.metrics = metrics
        # check if evaluation metric was provided
        if num_metric_to_set_weights is not None:
            self.evaluation_metric = self.metrics[num_metric_to_set_weights]
            self.metrics = tuple(metric for i, metric in enumerate(metrics) if i != num_metric_to_set_weights)
        else:
            self.evaluation_metric = None
        # If logger is provided, all the metrics during training process will be written
        self.logger = None

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = 0

    def print_and_log_metrics(self, metric_values: Tuple[float, ...],
                              eval_metric_value: Optional[float] = None) -> None:
        string_to_write = ''
        for metric_idx in range(len(self.metrics)):
            string_to_write += 'metric: %s, value:%f' % (self.metrics[metric_idx], metric_values[metric_idx])
        if eval_metric_value is not None:
            string_to_write += 'evaluation metric: %s, value:%f' % (self.evaluation_metric, eval_metric_value)
        print(string_to_write)
        # log it if logger is provided
        if self.logger is not None:
            self.logger.write(string_to_write + '\n')

    def on_epoch_end(self, epoch, logs=None):
        # TODO: write description
        metric_values, eval_metric_value = self.custom_recall_validation_with_generator()
        self.print_and_log_metrics(metric_values, eval_metric_value)
        # if evaluation_metric was chosen
        if self.evaluation_metric is not None:
            if np.greater(eval_metric_value, self.best):
                self.best = eval_metric_value
                self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        # TODO: write description
        if self.evaluation_metric is not None:
            self.model.set_weights(self.best_weights)

    def _get_ground_truth_and_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: write description
        total_predictions = np.zeros((0,))
        total_ground_truth = np.zeros((0,))
        for x, y in self.val_generator:
            predictions = model.predict(x)
            predictions = predictions.argmax(axis=-1).reshape((-1,))
            total_predictions = np.append(total_predictions, predictions)
            total_ground_truth = np.append(total_ground_truth, y.argmax(axis=-1).reshape((-1,)))
        return total_ground_truth, total_predictions

    def custom_recall_validation_with_generator(self) -> Tuple[Tuple[float, ...],
                                                               Union[float, None]]:
        # TODO: write description
        ground_truth, predictions = self._get_ground_truth_and_predictions()
        metric_values = []
        eval_metric_value = None
        for metric_idx in range(len(self.metrics)):
            metric_value = self.metrics[metric_idx](ground_truth, predictions)
            metric_values.append(metric_value)
        if self.evaluation_metric is not None:
            eval_metric_value = self.evaluation_metric(ground_truth, predictions)
        return tuple(metric_values), eval_metric_value


if __name__ == '__main__':
    a = get_annealing_LRreduce_callback(0.2, 0.1, 9)
    x = np.zeros((100, 100))
    y = np.ones((100, 1))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, activation='sigmoid', input_shape=(100,)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='binary_crossentropy')
    model.fit(x, y, epochs=5, batch_size=50, callbacks=[a])
    b = 1 + 2
