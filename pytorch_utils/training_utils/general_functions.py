import gc
from typing import Tuple, List, Optional

import torch

def train_step(model: torch.nn.Module, criterions: List[torch.nn.Module],
               inputs: List[torch.Tensor], ground_truth: List[torch.Tensor],
               device: torch.device) -> List[torch.Tensor]:
    """ Performs one training step for a model.

    :param model:
        model to train
    :param criterions:
        list of criterions to calculate loss
    :param inputs:
        list of input tensors
    :param ground_truth:
        list of ground truth tensors
    :param device:
        device to use for training

    :return: List[torch.Tensor]
        list of losses
    """
    # forward pass
    if len(inputs) == 1:
        inputs = inputs[0]
    outputs = model(inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    # checking input parameters
    if len(criterions) != len(outputs):
        raise ValueError("Number of criterions should be equal to number of outputs of the model.")
    # calculating criterions
    losses = []
    for criterion, output, gt in zip(criterions, outputs, ground_truth):
        losses.append(criterion(output, gt))
    # clear RAM from unused variables
    del outputs, ground_truth
    return losses


def train_epoch(model: torch.nn.Module, train_generator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterions: List[torch.nn.Module],
                device: torch.device, print_step: Optional[int] = 100,
                accumulate_gradients: Optional[int] = 1,
                batch_wise_lr_scheduller: Optional[object] = None,
                loss_multiplication_factor:Optional[float]=None) -> float:
    """ Performs one epoch of training for a model.

    :param model: torch.nn.Module
            Model to train.
    :param train_generator: torch.utils.data.DataLoader
            Generator for training data. Note that it should output the ground truths as a tuple of torch.Tensor
            (thus, we have several outputs).
    :param optimizer: torch.optim.Optimizer
            Optimizer for training.
    :param criterions: List[torch.nn.Module]
            Loss functions for each output of the model.
    :param device: torch.device
            Device to use for training.
    :param print_step: int
            Number of mini-batches between two prints of the running loss.
    :param accumulate_gradients: Optional[int]
            Number of mini-batches to accumulate gradients for. If 1, no accumulation is performed.
    :param batch_wise_lr_scheduller: Optional[torch.optim.lr_scheduler]
            Learning rate scheduller in case we have lr scheduller that executes the lr changing every mini-batch.
    :return: float
            Average loss for the epoch.
    """

    running_loss = 0.0
    total_loss = 0.0
    counter = 0
    for i, data in enumerate(train_generator):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(labels, list):
            labels = [labels]
        # move data to device
        inputs = [inp.float().to(device) for inp in inputs]
        labels = [lb.to(device) for lb in labels]

        # do train step
        with torch.set_grad_enabled(True):
            # form indecex of labels which should be one-hot encoded
            step_losses = train_step(model, criterions, inputs, labels, device)
            # normalize losses by number of accumulate gradient steps
            step_losses = [step_loss / accumulate_gradients for step_loss in step_losses]
            # backward pass
            sum_losses = sum(step_losses)
            if loss_multiplication_factor is not None:
                sum_losses = sum_losses * loss_multiplication_factor
            sum_losses.backward()
            # update weights if we have accumulated enough gradients
            if (i + 1) % accumulate_gradients == 0 or (i + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                if batch_wise_lr_scheduller is not None:
                    batch_wise_lr_scheduller.step()

        # print statistics
        running_loss += sum_losses.item()
        total_loss += sum_losses.item()
        counter += 1
        if i % print_step == (print_step - 1):  # print every print_step mini-batches
            print("Mini-batch: %i, loss: %.10f" % (i, running_loss / print_step))
            running_loss = 0.0
        # clear RAM from all the intermediate variables
        del inputs, labels, step_losses, sum_losses
    # clear RAM at the end of the epoch
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / counter