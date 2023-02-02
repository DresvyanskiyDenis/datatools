from typing import Optional

import dsntnn
import numpy as np
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
from torchinfo import summary

from pytorch_utils.models.Pose_estimation.MobileNetV2 import mobilenetv2_ed


class Modified_InceptionResnetV1(nn.Module):
    def __init__(self, dense_layer_neurons:int, num_classes:int, pretrained='vggface2', device:torch.device=None):
        super(Modified_InceptionResnetV1, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = InceptionResnetV1(pretrained=pretrained, classify=True, device=device)
        self.model.last_linear = nn.Identity()
        self.model.last_bn = nn.Identity()
        self.model.logits = nn.Identity()

        # new layers instead of old two last ones
        self.embeddings_layer = nn.Linear(1792, dense_layer_neurons)
        self.embeddings_batchnorm = nn.BatchNorm1d(dense_layer_neurons, momentum=0.1, affine=True)
        self.classifier = nn.Linear(dense_layer_neurons, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.embeddings_layer(x)
        x = self.embeddings_batchnorm(x)
        x = self.classifier(x)
        return x


class _Modified_MobileNetV2(nn.Module):
    def __init__(self, n_locations=16):
        super(_Modified_MobileNetV2, self).__init__()

        self.resnet = mobilenetv2_ed(width_mult=1.0)
        self.outsize = 32

        self.hm_conv = nn.Conv2d(self.outsize, n_locations, kernel_size=1, bias=False)


    def forward(self, images):
        # 1. Run the images through our Resnet
        backbone_model_out = self.resnet(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(backbone_model_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


class Modified_MobileNetV2_pose_estimation(nn.Module):
    def __init__(self, n_locations=16, pretrained:bool=True):
        super(Modified_MobileNetV2_pose_estimation, self).__init__()

        self.model = _Modified_MobileNetV2(n_locations=n_locations)
        path_to_weights = "/work/home/dsu/Model_weights/Pose_estimation/MobileNetV2/mobilenetv2_224_adam_best.t7"
        if pretrained:
            self.model.load_state_dict(torch.load(path_to_weights))

    def forward(self, x):
        x = self.model(x)
        return x






if __name__=='__main__':
    model = Modified_MobileNetV2_pose_estimation(n_locations=16, pretrained=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x = np.zeros((1,3,224,224))
    x = torch.from_numpy(x).float().to(device)
    y = model(x)










