from typing import Iterable, Tuple, Dict, Callable, Optional

import torch
import numpy as np
import os


class TorchEarlyStopping:

    def __init__(self, verbose: bool = True, patience: int = 5, save_path: str = "tmp"):
        self.verbose = verbose
        self.patience = patience
        self.save_path = save_path
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, epoch_loss: float, model: torch.nn.Module) -> bool:
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.counter = 0
            if self.verbose:
                print("Early stopping counter has been reset. Model is saved.")
            self._save_model(model)
            return False
        elif self.counter <= self.patience:
            self.counter += 1
            if self.verbose:
                print("Early stopping counter has been increased.")
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
                 need_argmax:bool=False):
        self.generator = generator
        self.model=model
        self.metrics = metrics
        self.device=device
        self.need_argmax=need_argmax

    def __call__(self) -> Dict[str, float]:
        with torch.no_grad():
            # create "general" arrays for real and predicted labels so that we can firstly predict labels (and save real labels)
            # and then calculate metrics once for the whole dataset
            predicted_labels = []
            real_labels = []
            for data, labels in self.generator:
                data, labels = data.to(self.device), labels.to(self.device)
                # forward pass
                outputs = self.model(data)
                # apply argmax to the outputs if needed
                if self.need_argmax:
                    outputs = torch.argmax(outputs, dim=-1)
                # append predicted labels and real labels to the "general" array
                predicted_labels.append(outputs.cpu().numpy().squeeze())
                real_labels.append(labels.cpu().numpy().squeeze())
            # flatten the arrays so that we can calculate metrics
            real_labels=np.array(real_labels).flatten()
            predicted_labels=np.array(predicted_labels).flatten()
            # calculate all defined metrics
            results = {}
            for metric_name, metric_func in self.metrics.items():
                results[metric_name] = metric_func(real_labels, predicted_labels)
            return results
