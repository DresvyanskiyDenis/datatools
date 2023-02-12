from typing import Iterable, Tuple, Dict, Callable, Optional, List

import torch
import numpy as np
import os

from torch.autograd import Variable
from torchinfo import summary

from decorators.common_decorators import timer


class TorchEarlyStopping:

    def __init__(self, mode:str='min', verbose: bool = True, patience: int = 5, save_path: str = "tmp"):
        self.verbose = verbose
        self.patience = patience
        self.save_path = save_path
        if mode=='min':
            self.best_loss = np.inf
            self.compare_operator= np.less
        else:
            self.best_loss = -np.inf
            self.compare_operator= np.greater
        self.counter = 0


    def __call__(self, epoch_loss: float, model: torch.nn.Module) -> bool:
        if self.compare_operator(epoch_loss, self.best_loss):
            self.best_loss = epoch_loss
            self.counter = 0
            if self.verbose:
                print("Early stopping counter has been reset. Model is saved.")
            self._save_model(model)
            return False
        elif self.counter <= self.patience:
            self.counter += 1
            if self.verbose:
                print("Early stopping counter has been increased. Current value: %d" % self.counter)
            return False
        else:
            if self.verbose:
                print(
                    f"Early stopping counter has reached patience level: {self.patience}. Stopping the training process...")
            return True

    def _save_model(self, model: torch.nn.Module):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
        print(f"Saved model to {self.save_path}")


class TorchMetricEvaluator:

    def __init__(self, generator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                 model: torch.nn.Module,
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
                 device: torch.device,
                 output_argmax:bool=False,
                 output_softmax:bool=False,
                 labels_argmax:bool=False,
                 loss_func:Optional=None,
                 separate_inputs:bool=False):
        self.generator = generator
        self.model=model
        self.metrics = metrics
        self.device=device
        self.output_argmax=output_argmax
        self.output_softmax=output_softmax
        self.labels_argmax = labels_argmax
        self.loss_func=loss_func
        self.separate_inputs=separate_inputs

    @timer
    def __call__(self) -> Dict[str, float]:
        with torch.no_grad():
            # turn model to the evaluation mode
            self.model.eval()
            # create "general" arrays for real and predicted labels so that we can firstly predict labels (and save real labels)
            # and then calculate metrics once for the whole dataset
            predicted_labels = []
            real_labels = []
            # if needed to return loss
            loss=0
            counter=0
            for data, labels in self.generator:
                # transform labels to the 1D array with long type if they are just digits (numbers of classes)
                if not self.labels_argmax:
                    labels = torch.squeeze(labels).long()
                if self.separate_inputs:
                    data = [Variable(x.float()) for x in data]
                    data = [x.to(self.device) for x in data]
                else:
                    data = data.float()
                    data = data.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                if self.separate_inputs:
                    outputs = self.model(*data)
                else:
                    outputs = self.model(data)
                if self.loss_func is not None:
                    loss+=self.loss_func(outputs, labels).item()
                    counter+=1
                # apply softmax to the outputs if needed
                if self.output_softmax:
                    outputs = torch.softmax(outputs, dim=-1)
                # apply argmax to the outputs if needed
                if self.output_argmax:
                    outputs = torch.argmax(outputs, dim=-1)
                # apply argmax to the labels if needed
                if self.labels_argmax:
                    labels = torch.argmax(labels, dim=-1)
                # append predicted labels and real labels to the "general" array
                predicted_labels.append(outputs.cpu().numpy().squeeze())
                real_labels.append(labels.cpu().numpy().squeeze())
            # check the lists on the elements with 0 len shapes
            predicted_labels = self._batch_check(predicted_labels)
            real_labels = self._batch_check(real_labels)
            # flatten the arrays so that we can calculate metrics
            real_labels=np.concatenate(real_labels).flatten()
            predicted_labels=np.concatenate(predicted_labels).flatten()
            # calculate all defined metrics
            results = {}
            for metric_name, metric_func in self.metrics.items():
                results[metric_name] = metric_func(real_labels, predicted_labels)
            # calculate total loss if specified
            if self.loss_func is not None:
                loss/=counter
                results['loss']=loss
        # turn model back to the train mode
        self.model.train()
        return results

    def _batch_check(self, array):
        # there are errors when the last batch of the dataset contains just one instance
        # the error is that after the processing by model, the output of the model is one-element array,
        # which automatically converts to the np-array with len of shape = 0
        # therefore, when concatenation of the outputs happens, the np.concatenate() function
        # cannot concat arrays with len of arrays shape equals 1 and 0 (at the end of the list)
        # therefore, we need to check if the list contains arrays with 0 len shape and fix it by increasing it by one
        for i, element in enumerate(array):
            if len(element.shape) == 0:
                array[i] = np.expand_dims(element, axis=0)
        return array


class GradualLayersUnfreezer():

    def __init__(self, model:torch.nn.Module, layers:List[torch.nn.Module],
                 layers_per_epoch:int,
                 layers_to_unfreeze_before_start:Optional[int]=None,
                 verbose:bool=False, input_shape:Optional[Tuple[int, ...]]=None):
        """ Gradually unfreezes layers of the model with every __call__.
        The layers are unfrozen in the following order: the last layer, the second last layer, etc.

        :param model: torch.nn.Module
                Pytorch model, which layers should be unfrozen over time
        :param layers: List[torch.nn.Module]
                List of the layers, which should be unfrozen over time
        :param layers_per_epoch: int
                Number of the layers, which should be unfrozen in every epoch
        :param layers_to_unfreeze_before_start: int
                Number of the layer (as an int number), which should be unfrozen before the training process starts
        :param verbose: bool
                If True, prints the information about the layers, which are unfrozen as well as the model structure
        :param input_shape: tuple
                Shape of the input data. It is needed to create the dummy input for the model, if the verbose is True.
        """
        self.model=model
        self.layers = layers
        self.layers_per_epoch = layers_per_epoch
        self.verbose=verbose
        self.layers_to_unfreeze_before_start = layers_to_unfreeze_before_start
        self.current_last_unfrozen_layer = len(self.layers)
        self.input_shape = input_shape

        # firstly, freeze all layers
        self._freeze_all_layers()
        # then, unfreeze layers, which should be unfrozen before the training process starts
        if self.layers_to_unfreeze_before_start is not None:
            self._unfreeze_next_layers(self.layers_to_unfreeze_before_start)
            if self.verbose:
                print("Current model structure and the number of params, which are unfrozen.")
                summary(self.model, input_size=self.input_shape)



    def __call__(self):
        """
         Unfreezes next self.layers_per_epoch layers of the model. It starts from the last layers and go
         down to the first layers
        :return: None
        """
        # check if we need to unfreeze the next layers
        if self.current_last_unfrozen_layer > 0:
            # unfreeze the next layers
            print("Layer names:")
            for num_layer, child in enumerate(self.layers):
                if num_layer<self.current_last_unfrozen_layer and num_layer>=self.current_last_unfrozen_layer - self.layers_per_epoch:
                    print(child)
            self._unfreeze_next_layers(self.layers_per_epoch)
            if self.verbose:
                print("Current model structure and trainable parameters:")
                summary(self.model, input_size=self.input_shape)

        else:
            print("No layers has been unfrozen, since all layers are already unfrozen")


    def _freeze_all_layers(self):
        for child in self.layers:
            for param in child.parameters():
                param.requires_grad = False
                param.requires_grad_(False)


    def _unfreeze_next_layers(self, num_layers:int):
        min_num_layer = max(0, self.current_last_unfrozen_layer - num_layers)
        max_num_layer = self.current_last_unfrozen_layer
        for num_layer, child in enumerate(self.layers):
            # check if we are still in the range of layers, which should be unfrozen
            if num_layer <= max_num_layer and num_layer >= min_num_layer:
                if self.verbose:
                    print(f"Unfreezing layer {child}...")
                # unfreeze the layer
                for param in child.parameters():
                    param.requires_grad = True
                    param.requires_grad_(True)
        # update the number of the last unfrozen layer
        self.current_last_unfrozen_layer = min_num_layer