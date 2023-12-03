from functools import partial
from typing import Optional, Sequence

import torch
from audtorch.metrics.functional import concordance_cc
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchmetrics import ConcordanceCorrCoef


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:


        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SoftFocalLoss(nn.Module):
    # focal loss for soft targets, i. e. target can be [0, 0.3, 0.7, 1]
    # class FocalLoss takes only digit targets
    def __init__(self,
                 softmax: bool,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.softmax = softmax

    def __repr__(self):
        arg_keys = ['alpha', 'gamma']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        if self.softmax:
            p = F.softmax(x, dim=-1)
        else:
            p = x

        epsilon = 1e-7
        p = torch.clip(p, epsilon, 1. - epsilon)

        cross_entropy = -y * torch.log(p)

        # focal loss
        loss = self.alpha * torch.pow(1. - p, self.gamma) * cross_entropy

        loss = torch.sum(loss, dim =-1).mean()

        return loss


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return 1 - self.ccc(yhat, y)

    def ccc(self, outputs, labels):
        labels_mean = torch.mean(labels, dim=-1, keepdim=True)
        labels_var = torch.mean(torch.square(labels - labels_mean), dim=-1, keepdim=True)

        outputs_mean = torch.mean(outputs, dim=-1, keepdim=True)
        outputs_var = torch.mean(torch.square(outputs - outputs_mean), dim=-1, keepdim=True)

        cov = torch.mean((labels - labels_mean) * (outputs - outputs_mean), dim=-1, keepdim=True)

        ccc = (2.0 * cov) / (outputs_var + labels_var + torch.square(labels_mean - outputs_mean) + 1e-7)
        return torch.mean(ccc)


if __name__=="__main__":
    import numpy as np
    focal_loss=SoftFocalLoss(softmax=True, alpha=torch.Tensor([0.5, 0.3, 0.2]), gamma=2)
    y_true=torch.Tensor(np.array([[0,0,1],
                     [1,0,0],
                     [1,0,0],
                     [0,1,0],
                     [0,0,1]
                     ]))
    predictions=torch.Tensor(np.array([[0,0.35,0.65],
                     [0.3,0.6,0.1],
                     [0.7,0.3,0],
                     [0.1,0.54,0.36],
                     [0,0,1]
                     ]))
    print(focal_loss(predictions, y_true))

    focal_loss = FocalLoss(alpha=torch.Tensor([0.5, 0.3, 0.2]), gamma=2)
    y_true = torch.Tensor([2,0,0,1,2])
    y_pred = torch.Tensor(np.array([[0,0.35,0.65],
                     [0.3,0.6,0.1],
                     [0.7,0.3,0],
                     [0.1,0.54,0.36],
                     [0,0,1]
                     ]))

    print(focal_loss(predictions, y_true.long()))