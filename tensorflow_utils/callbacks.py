from functools import partial
from typing import Callable, Iterable, Tuple

import tensorflow as tf
import numpy as np

def get_reduceLRonPlateau_callback(monitoring_loss:str='val_loss', reduce_factor:float=0.1,
                                   num_patient_epochs:int=10, min_lr:float=0.)->tf.keras.callbacks.ReduceLROnPlateau:
    # TODO: write description
    callback=tf.keras.callbacks.ReduceLROnPlateau(monitor=monitoring_loss, factor=reduce_factor,
                                                  patience=num_patient_epochs, min_lr=min_lr)
    return callback

def get_LRreduce_callback(scheduller:Callable[[int, float], float])->tf.keras.callbacks.LearningRateScheduler:
    #TODO: write description
    callback = tf.keras.callbacks.LearningRateScheduler(scheduller)
    return callback

def get_annealing_LRreduce_callback(highest_lr:float,lowest_lr:float, annealing_period:int)->tf.keras.callbacks.LearningRateScheduler:
    # TODO: write description
    learning_rates=np.linspace(start=highest_lr, stop=lowest_lr, num=annealing_period)
    def get_lr_according_to_annealing_step(current_epoch:int, current_lr:float, learning_rates:np.ndarray)->float:
        annealing_period=learning_rates.shape[0]
        idx_of_annealing_step=current_epoch%annealing_period
        return learning_rates[idx_of_annealing_step]
    annealing_func_for_tf_callback=partial(get_lr_according_to_annealing_step, learning_rates=learning_rates)
    callback=get_LRreduce_callback(annealing_func_for_tf_callback)
    return callback

class best_weights_setter_callback(tf.keras.callbacks.Callback):
    """Calculates the recall score at the end of each training epoch and saves the best weights across all the training
        process. At the end of training process, it will set weights of the model to the best found ones.
        # TODO: write, which types of metric functions it supports
    """
    def __init__(self, val_generator: Iterable[Tuple[np.ndarray, np.ndarray]],
                 evaluation_metric:Callable[[np.ndarray, np.ndarray], float]):
        super(best_weights_setter_callback, self).__init__()
        # best_weights to store the weights at which the minimum UAR occurs.
        self.best_weights = None
        # generator to iterate on it on every end of epoch
        self.val_generator = val_generator
        self.evaluation_metric=evaluation_metric

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        # TODO: write description
        current_recall = self.custom_recall_validation_with_generator(self.val_generator, self.model)
        print('current validation %s:'%self.evaluation_metric, current_recall)
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




if __name__=='__main__':

    a=get_annealing_LRreduce_callback(0.2, 0.1, 9)
    x=np.zeros((100, 100))
    y=np.ones((100,1))
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, activation='sigmoid', input_shape=(100,)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='binary_crossentropy')
    model.fit(x,y, epochs=5, batch_size=50, callbacks=[a])
    b=1+2
