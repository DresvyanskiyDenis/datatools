#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import Tuple, Optional, Union

import torch


class DenseModel(torch.nn.Module):
    activation_functions_mapping = {'relu': torch.nn.ReLU,
                                    'sigmoid': torch.nn.Sigmoid,
                                    'tanh': torch.nn.Tanh,
                                    'softmax': torch.nn.Softmax,
                                    'linear': torch.nn.Linear
                                    }

    def __init__(self, input_shape:int, dense_neurons: Tuple[int,...], activations:Union[str,Tuple[str,...]]='relu',
                    dropout: Optional[float] = 0.3,
                    regularization:Optional[bool]=None,
                    output_neurons: Union[Tuple[int,...], int] = 7,
                    activation_function_for_output:str='softmax'):
        super(DenseModel, self).__init__()
        self.input_shape = input_shape
        self.dense_neurons = dense_neurons
        self.activations = activations
        self.dropout = dropout
        self.regularization = regularization
        self.output_neurons = output_neurons
        self.activation_function_for_output = activation_function_for_output
        # build the model
        self._build_model()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def _build_model(self):
        self.layers=[]
        # input layer
        self.layers.append(torch.nn.Linear(self.input_shape, self.dense_neurons[0]))
        self.layers.append(self.activation_functions_mapping[self.activations[0]]())
        # hidden layers
        for i in range(len(self.dense_neurons)-1):
            self.layers.append(torch.nn.Linear(self.dense_neurons[i], self.dense_neurons[i+1]))
            self.layers.append(self.activation_functions_mapping[self.activations[i+1]]())
        # output layer
        self.layers.append(torch.nn.Linear(self.dense_neurons[-1], self.output_neurons))
        self.layers.append(self.activation_functions_mapping[self.activation_function_for_output]())



if __name__ == '__main__':
    model = DenseModel(input_shape=100, dense_neurons=(200, 300, 500), activations=('relu', 'sigmoid', 'tanh'),
                    dropout= 0.3,
                    regularization=None,
                    output_neurons= 4,
                    activation_function_for_output='softmax')

    print('model structure:')
    print(model)