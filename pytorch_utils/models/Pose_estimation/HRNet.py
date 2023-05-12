from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from SimpleHRNet import SimpleHRNet


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
                # block 1
                ("conv1_new", torch.nn.Conv2d(13, 128, kernel_size=(3, 3), stride=(1, 1), padding="same") if self.consider_only_upper_body
                 else torch.nn.Conv2d(17, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
                 ),
                ("dropout1_new", torch.nn.Dropout(0.1)),
                ("BatchNormalization1_new", torch.nn.BatchNorm2d(128)),
                ("relu1_new", torch.nn.ReLU()),
                ("maxpool1_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 64x64
                # block 2
                ("conv2_new", torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("dropout2_new", torch.nn.Dropout(0.1)),
                ("BatchNormalization2_new", torch.nn.BatchNorm2d(128)),
                ("relu2_new", torch.nn.ReLU()),
                ("maxpool2_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 32x32
                # block 3
                ("conv3_new", torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("dropout3_new", torch.nn.Dropout(0.1)),
                ("BatchNormalization3_new", torch.nn.BatchNorm2d(256)),
                ("relu3_new", torch.nn.ReLU()),
                ("maxpool3_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16
                # block 4
                ("conv4_new", torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding="same")),
                ("dropout4_new", torch.nn.Dropout(0.1)),
                ("BatchNormalization4_new", torch.nn.BatchNorm2d(256)),
                ("relu4_new", torch.nn.ReLU()),
                ("maxpool4_new", torch.nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8
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