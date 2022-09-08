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
                                    'elu': torch.nn.ELU,
                                    'leaky_relu': torch.nn.LeakyReLU,
                                    'linear': None
                                    }

    def __init__(self, input_shape:int, dense_neurons: Tuple[int,...], activations:Union[str,Tuple[str,...]]='relu',
                    dropout: Optional[float] = None,
                    output_neurons: Union[Tuple[int,...], int] = 7,
                    activation_function_for_output:str='softmax'):
        super(DenseModel, self).__init__()
        self.input_shape = input_shape
        self.dense_neurons = dense_neurons
        self.activations = activations
        self.dropout = dropout
        self.output_neurons = output_neurons
        self.activation_function_for_output = activation_function_for_output
        # build the model
        self._build_model()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def _build_model(self):
        self.layers=torch.nn.ModuleList()
        # input layer
        self.layers.append(torch.nn.Linear(self.input_shape, self.dense_neurons[0]))
        self.layers.append(self.activation_functions_mapping[self.activations[0]]())
        # hidden layers
        for i in range(len(self.dense_neurons)-1):
            self.layers.append(torch.nn.Linear(self.dense_neurons[i], self.dense_neurons[i+1]))
            self.layers.append(self.activation_functions_mapping[self.activations[i+1]]())
            if self.dropout:
                self.layers.append(torch.nn.Dropout(self.dropout))
        # output layer
        self.layers.append(torch.nn.Linear(self.dense_neurons[-1], self.output_neurons))
        if self.activation_function_for_output == 'softmax':
            self.layers.append(torch.nn.Softmax(dim=-1))
        elif self.activation_function_for_output == 'linear':
            pass
        else:
            self.layers.append(self.activation_functions_mapping[self.activation_function_for_output]())
