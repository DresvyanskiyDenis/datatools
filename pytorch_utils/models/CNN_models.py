from typing import Optional

import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
from torchinfo import summary
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, EfficientNet_B1_Weights, efficientnet_b1, \
    EfficientNet_B4_Weights, efficientnet_b4, ViT_B_16_Weights, vit_b_16

from pytorch_utils.models.Pose_estimation import MobileNetV2


class Modified_InceptionResnetV1(nn.Module):
    def __init__(self, dense_layer_neurons:int, num_classes:Optional[int], pretrained='vggface2', device:torch.device=None):
        super(Modified_InceptionResnetV1, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = InceptionResnetV1(pretrained=pretrained, classify=True, device=device)
        self.model.last_linear = nn.Identity()
        self.model.last_bn = nn.Identity()
        self.model.logits = nn.Identity()

        # new layers instead of old two last ones
        self.dropout_after_pretrained = nn.Dropout(0.1)
        self.embeddings_layer = nn.Linear(1792, dense_layer_neurons)
        self.embeddings_batchnorm = nn.BatchNorm1d(dense_layer_neurons, momentum=0.1, affine=True)
        self.activation_embeddings_layer = nn.Tanh()
        if num_classes:
            self.classifier = nn.Linear(dense_layer_neurons, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout_after_pretrained(x)
        x = self.embeddings_layer(x)
        x = self.embeddings_batchnorm(x)
        x = self.activation_embeddings_layer(x)
        if self.num_classes:
            x = self.classifier(x)
        return x


class _Pose_Estimation_MobileNetV2(nn.Module):
    def __init__(self, n_locations=16):
        super(_Pose_Estimation_MobileNetV2, self).__init__()

        self.resnet = MobileNetV2.mobilenetv2_ed(width_mult=1.0)
        self.outsize = 32
        self.hm_conv = nn.Conv2d(self.outsize, n_locations, kernel_size=1, bias=False)


    def forward(self, images):
        # 1. Run the images through our Resnet
        backbone_model_out = self.resnet(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(backbone_model_out)
        # 3. Normalize the heatmaps
        #heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        #coords = dsntnn.dsnt(heatmaps)

        return unnormalized_heatmaps


class Modified_MobileNetV2_pose_estimation(nn.Module):
    def __init__(self, n_locations=16, pretrained:bool=True, embeddings_layer_neurons:int=256):
        super(Modified_MobileNetV2_pose_estimation, self).__init__()

        self.model = _Pose_Estimation_MobileNetV2(n_locations=n_locations)
        self.embeddings_layer_neurons = embeddings_layer_neurons
        path_to_weights = "/work/home/dsu/Model_weights/Pose_estimation/MobileNetV2/mobilenetv2_224_adam_best.t7"
        if pretrained:
            self.model.load_state_dict(torch.load(path_to_weights))
        # stack 2D conv layers upon heatmaps
        self.stacked_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_locations, out_channels=128, kernel_size=3, padding=1), # 128x56x56
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4), # 128x14x14

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # 256x14x14
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)) # 256x1x1
        )
        self.dropout_after_pretrained = nn.Dropout(0.1)
        self.embeddings_layer = nn.Linear(256, embeddings_layer_neurons)
        self.batchnorm = nn.BatchNorm1d(embeddings_layer_neurons, momentum=0.1, affine=True)
        self.activation_embeddings = nn.Tanh()


    def forward(self, x):
        heatmaps = self.model(x)
        x = self.stacked_conv(heatmaps)
        x = x.view(x.size(0), -1)
        x = self.dropout_after_pretrained(x)
        x = self.embeddings_layer(x)
        x = self.batchnorm(x)
        x = self.activation_embeddings(x)
        return x


class Modified_MobileNetV3_large(nn.Module):

    def __init__(self,pretrained:bool=True, embeddings_layer_neurons:int=256,
                 num_classes:Optional[int]=None, num_regression_neurons:Optional[int]=None):
        super(Modified_MobileNetV3_large, self).__init__()

        self.pretrained = pretrained
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.num_regression_neurons = num_regression_neurons
        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        else:
            weights = None
        self.model = mobilenet_v3_large(weights=weights)
        self.model.classifier = nn.Identity()
        self.dropout_after_pretrained = nn.Dropout(0.1)
        self.embeddings_layer = nn.Linear(960, embeddings_layer_neurons)
        self.batchnorm = nn.BatchNorm1d(embeddings_layer_neurons, momentum=0.1, affine=True)
        self.activation_embeddings = nn.Tanh()
        if num_classes:
            self.classifier = nn.Linear(embeddings_layer_neurons, num_classes)
        if num_regression_neurons:
            self.regression = nn.Linear(embeddings_layer_neurons, num_regression_neurons)


    def forward(self, x):
        x = self.model(x)
        x = self.dropout_after_pretrained(x)
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

class Modified_EfficientNet_B1(nn.Module):

    def __init__(self, pretrained:bool=True, embeddings_layer_neurons:int=256,
                 num_classes:Optional[int]=None, num_regression_neurons:Optional[int]=None):
        super(Modified_EfficientNet_B1, self).__init__()

        self.pretrained = pretrained
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.num_regression_neurons = num_regression_neurons
        if pretrained:
            weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        else:
            weights = None
        self.model = efficientnet_b1(weights=weights)
        self.model.classifier = nn.Identity()
        self.dropout_after_pretrained = nn.Dropout(0.1)
        self.embeddings_layer = nn.Linear(1280, embeddings_layer_neurons)
        self.batchnorm = nn.BatchNorm1d(embeddings_layer_neurons, momentum=0.1, affine=True)
        self.activation_embeddings = nn.Tanh()
        if num_classes:
            self.classifier = nn.Linear(embeddings_layer_neurons, num_classes)
        if num_regression_neurons:
            self.regression = nn.Linear(embeddings_layer_neurons, num_regression_neurons)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout_after_pretrained(x)
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


class Modified_EfficientNet_B4(nn.Module):

    def __init__(self, pretrained:bool=True, embeddings_layer_neurons:int=256,
                 num_classes:Optional[int]=None, num_regression_neurons:Optional[int]=None):
        super(Modified_EfficientNet_B4, self).__init__()

        self.pretrained = pretrained
        self.embeddings_layer_neurons = embeddings_layer_neurons
        self.num_classes = num_classes
        self.num_regression_neurons = num_regression_neurons
        if pretrained:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = efficientnet_b4(weights=weights)
        self.model.classifier = nn.Identity()
        self.dropout_after_pretrained = nn.Dropout(0.1)
        self.embeddings_layer = nn.Linear(1792, embeddings_layer_neurons)
        self.batchnorm = nn.BatchNorm1d(embeddings_layer_neurons, momentum=0.1, affine=True)
        self.activation_embeddings = nn.Tanh()
        if num_classes:
            self.classifier = nn.Linear(embeddings_layer_neurons, num_classes)
        if num_regression_neurons:
            self.regression = nn.Linear(embeddings_layer_neurons, num_regression_neurons)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout_after_pretrained(x)
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

class Modified_ViT_B_16(nn.Module):

        def __init__(self, pretrained:bool=True, embeddings_layer_neurons:int=256,
                    num_classes:Optional[int]=None, num_regression_neurons:Optional[int]=None):
            super(Modified_ViT_B_16, self).__init__()

            self.pretrained = pretrained
            self.embeddings_layer_neurons = embeddings_layer_neurons
            self.num_classes = num_classes
            self.num_regression_neurons = num_regression_neurons
            if pretrained:
                weights = ViT_B_16_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.model = vit_b_16(weights=weights)
            self.model.classifier = nn.Identity()
            self.dropout_after_pretrained = nn.Dropout(0.1)
            self.embeddings_layer = nn.Linear(1000, embeddings_layer_neurons)
            self.batchnorm = nn.BatchNorm1d(embeddings_layer_neurons, momentum=0.1, affine=True)
            self.activation_embeddings = nn.Tanh()
            if num_classes:
                self.classifier = nn.Linear(embeddings_layer_neurons, num_classes)
            if num_regression_neurons:
                self.regression = nn.Linear(embeddings_layer_neurons, num_regression_neurons)

        def forward(self, x):
            x = self.model(x)
            x = self.dropout_after_pretrained(x)
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









if __name__=='__main__':
    model = Modified_ViT_B_16(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    device = torch.device('cpu')
    model.to(device)
    summary(model, input_size=(2,3,224,224), device=device)


