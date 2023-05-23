from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from SimpleHRNet import SimpleHRNet
from pytorch_utils.layers.conv_layers import ResidualBlock


def _load_HRNet_model(device:str,
    path_to_weights: str) -> torch.nn.Module:

    model = SimpleHRNet(c=32, nof_joints=17,
                        checkpoint_path=path_to_weights, multiperson=False,
                        return_heatmaps=False, return_bounding_boxes=False,
                        device=device)
    model = model.model
    return model


class Modified_HRNet(nn.Module):

    def __init__(self, pretrained: bool = True, path_to_weights:Optional[str]=None, embeddings_layer_neurons: int = 256,
                 num_classes: Optional[int] = None, num_regression_neurons: Optional[int] = None,
                 consider_only_upper_body: bool = False):
        super(Modified_HRNet, self).__init__()

        self.pretrained = pretrained
        self.path_to_weights = path_to_weights
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.num_regression_neurons = num_regression_neurons
        self.consider_only_upper_body = consider_only_upper_body
        if pretrained:
            if self.path_to_weights is None:
                raise ValueError('You must provide path to weights')
            else:
                self.HRNet = _load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                             path_to_weights = self.path_to_weights)
        else:
            self.HRNet = SimpleHRNet(c=32, nof_joints=17, checkpoint_path=None, multiperson=False,
                        return_heatmaps=False, return_bounding_boxes=False, device=device)

        # "cut off" last layer
        self.HRNet.final_layer = torch.nn.Identity()
        # freeze HRNet parts
        self._freeze_hrnet_parts()
        # build additional layers
        self._build_additional_layers()
        # build embedding layers
        self.dropout_after_conv = nn.Dropout(0.1)
        self.embeddings_layer = nn.Linear(256, embeddings_layer_neurons)
        self.batchnorm = nn.BatchNorm1d(embeddings_layer_neurons, momentum=0.1, affine=True)
        self.activation_embeddings = nn.Tanh()
        if num_classes:
            self.classifier = nn.Linear(embeddings_layer_neurons, num_classes)
        if num_regression_neurons:
            self.regression = nn.Linear(embeddings_layer_neurons, num_regression_neurons)

    def _freeze_hrnet_parts(self):
        for name, param in self.HRNet.named_parameters():
            param.requires_grad = False

    def _build_additional_layers(self):
        self.additional_layers = torch.nn.Sequential(
            OrderedDict([
                # Residual block 1
                ("residual_block_1", ResidualBlock(13, 128, stride=2, downsample=True)),
                # Residual block 2
                ("residual_block_2", ResidualBlock(128, 128, stride=2, downsample=True)),
                # Residual block 3
                ("residual_block_3", ResidualBlock(128, 256, stride=2, downsample=True)),
                # Global avg pool
                ("globalpool_new", torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))),
                # Flatten
                ("flatten_new", torch.nn.Flatten()),
            ]
            )
        )


    def forward(self, x):
        x = self.HRNet(x)
        # if we consider only upper body, we need to cut off the lower body (joints 13-16, numebred from 0)
        if self.consider_only_upper_body:
            x = x[:, 0:13, :, :]
        x = self.additional_layers(x)
        x = self.dropout_after_conv(x)
        x = self.embeddings_layer(x)
        x = self.batchnorm(x)
        x = self.activation_embeddings(x)
        output = ()

        if self.num_classes:
            x_classification = self.classifier(x)
            output += (x_classification,)
        if self.num_regression_neurons:
            x_regression = self.regression(x)
            output += (x_regression,)

        if len(output) == 0:
            return x
        elif len(output) == 1:
            return output[0]
        else:
            return output