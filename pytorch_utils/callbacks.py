from typing import Iterable, Tuple, Dict, Callable, Optional

import torch
import numpy as np
import os


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
        torch.save(model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
        print(f"Saved model to {self.save_path}")


class TorchMetricEvaluator:

    def __init__(self, generator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                 model: torch.nn.Module,
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
                 device: torch.device,
                 need_argmax:bool=False,
                 need_softmax:bool=False,
                 loss_func:Optional=None):
        self.generator = generator
        self.model=model
        self.metrics = metrics
        self.device=device
        self.need_argmax=need_argmax
        self.need_softmax=need_softmax
        self.loss_func=loss_func

    def __call__(self) -> Dict[str, float]:
        with torch.no_grad():
            # create "general" arrays for real and predicted labels so that we can firstly predict labels (and save real labels)
            # and then calculate metrics once for the whole dataset
            predicted_labels = []
            real_labels = []
            # if needed to return loss
            loss=0
            counter=0
            for data, labels in self.generator:
                labels = torch.squeeze(labels).long()
                data, labels = data.to(self.device), labels.to(self.device)
                # forward pass
                outputs = self.model(data)
                if self.loss_func is not None:
                    loss+=self.loss_func(outputs, labels).item()
                    counter+=1
                # apply softmax to the outputs if needed
                if self.need_softmax:
                    outputs = torch.softmax(outputs, dim=-1)
                # apply argmax to the outputs if needed
                if self.need_argmax:
                    outputs = torch.argmax(outputs, dim=-1)
                # append predicted labels and real labels to the "general" array
                predicted_labels.append(outputs.cpu().numpy().squeeze())
                real_labels.append(labels.cpu().numpy().squeeze())
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
            return results
