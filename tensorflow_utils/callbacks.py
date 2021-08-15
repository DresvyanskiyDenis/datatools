#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains the functions for keras callbacks creation.

List of functions:

    * get_reduceLRonPlateau_callback - creates a reduceLRonPlateau keras callback with defined parameters.
    * get_LRreduce_callback - creates a LearningRateScheduler keras callback based on the provided Callable function.
    * get_annealing_LRreduce_callback - creates a LearningRateScheduler, which imitates the annealing learning rate reduce.

List of classes:

    * best_weights_setter_callback - saves the weights of best model, monitoring the defined metric on validation generator.
"""

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from functools import partial
from typing import Callable, Iterable, Tuple

import tensorflow as tf
import numpy as np


def get_reduceLRonPlateau_callback(monitoring_loss: str = 'val_loss', reduce_factor: float = 0.1,
                                   num_patient_epochs: int = 10,
                                   min_lr: float = 0.) -> tf.keras.callbacks.ReduceLROnPlateau:
    """ Creates the reduceLRonPlateau callback done before by keras library.
    see: https://keras.io/api/callbacks/reduce_lr_on_plateau/

    :param monitoring_loss: str
            the name of loss to monitor.
    :param reduce_factor: float
            the factor on which the learning rate will be reduced after waiting defined epochs without improvements.
    :param num_patient_epochs: int
            the number of epochs before the learning rate will be reduced.
    :param min_lr: float
            the minimal value of the learning rate can be.
    :return: tf.keras.callbacks.ReduceLROnPlateau
            created callback for the tf.keras.Model
    """
    callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitoring_loss, factor=reduce_factor,
                                                    patience=num_patient_epochs, min_lr=min_lr)
    return callback


def get_LRreduce_callback(scheduller: Callable[[int, float], float]) -> tf.keras.callbacks.LearningRateScheduler:
    """ Creates a LearningRateScheduler keras callback based on the provided Callable function.
    See: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

    :param scheduller: Callable[[int, float], float]
            function, which takes current epoch and learning rate and returns a new learning rate
    :return: tf.keras.callbacks.LearningRateScheduler
            LearningRateScheduler
    """
    callback = tf.keras.callbacks.LearningRateScheduler(scheduller)
    return callback


def get_annealing_LRreduce_callback(highest_lr: float, lowest_lr: float,
                                    annealing_period: int) -> tf.keras.callbacks.LearningRateScheduler:
    """Creates the annealing LRreduce callback based on the tf.keras.callbacks.LearningRateScheduler.

    :param highest_lr: float
            the highest value of the learning rate
    :param lowest_lr: float
            the lowest value of the learning rate
    :param annealing_period: int
            the number of epochs, on which the learning rate will be gradually decreased
    :return: tf.keras.callbacks.LearningRateScheduler
            The tf.keras.callbacks.LearningRateScheduler, which imitates the annealing learning rate reduce strategy
    """
    learning_rates = np.linspace(start=highest_lr, stop=lowest_lr, num=annealing_period)

    def get_lr_according_to_annealing_step(current_epoch: int, learning_rates: np.ndarray) -> float:
        annealing_period = learning_rates.shape[0]
        idx_of_annealing_step = current_epoch % annealing_period
        return learning_rates[idx_of_annealing_step]

    annealing_func_for_tf_callback = partial(get_lr_according_to_annealing_step, learning_rates=learning_rates)
    callback = get_LRreduce_callback(annealing_func_for_tf_callback)
    return callback


class best_weights_setter_callback(tf.keras.callbacks.Callback):
    """Saves the weights of the best performed model with chosen metric.
       Weights will be set at the end of training.

    """

    def __init__(self, val_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                 evaluation_metric: Callable[[np.ndarray, np.ndarray], float]):
        super(best_weights_setter_callback, self).__init__()
        """
        :param val_generator: Iterable[Tuple[np.ndarray, np.ndarray]]
            The generator, which returns the tuple of ()data, labels)
        :param evaluation_metric: Callable[[np.ndarray, np.ndarray], float]
            The function (metric), which takes Tuple of np.arrays (ground_truth, predictions) and gives the float value,
            in the other words, the metric score (from 0 to 1).
        """
        # best_weights to store the weights at which the minimum UAR occurs.
        self.best_weights = None
        # generator to iterate on it on every end of epoch
        self.val_generator = val_generator
        self.evaluation_metric = evaluation_metric

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        """ Do defined action on the end of training epoch.

        :param epoch: int
                the current number of epoch.
        :param logs:
                logs to write and get from them information.
        :return: None
        """
        current_recall = self.custom_validation_with_generator(self.val_generator, self.model)
        print('current validation %s:' % self.evaluation_metric, current_recall)
        if np.greater(current_recall, self.best):
            self.best = current_recall
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        """Do defined action on the end of training procedure.

        :param logs:
                logs to write and get from them information.
        :return: None
        """
        self.model.set_weights(self.best_weights)

    def custom_validation_with_generator(self, generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                                                model: tf.keras.Model) -> float:
        """ Evaluates the model with provided on initialization generator.
            The chosen metric will be used for the evaluation.

        :param generator: Iterable[Tuple[np.ndarray, np.ndarray]]
                The generator, which returns the tuple of (data, labels)
        :param model: tf.keras.Model
                The keras model, which will be evaluated
        :return: float
                The value of the metric
        """
        total_predictions = np.zeros((0,))
        total_ground_truth = np.zeros((0,))
        for x, y in generator:
            predictions = model.predict(x)
            predictions = predictions.argmax(axis=-1).reshape((-1,))
            total_predictions = np.append(total_predictions, predictions)
            total_ground_truth = np.append(total_ground_truth, y.argmax(axis=-1).reshape((-1,)))
        return self.evaluation_metric(total_ground_truth, total_predictions)

