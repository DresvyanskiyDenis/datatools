from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms import functional as F

class EfficientNet_image_preprocessor(nn.Module):
    """The class taken from the torchvision library for image preprocesing in case of using EfficientNet models.
    The link to GitHub is: https://github.com/pytorch/vision
    It should be noted, that the forward function has been modified:
    The central cropping as well as resizing are omitted"""
    def __init__(self) -> None:
        super().__init__()
        self.mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
        self.std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    def forward(self, img: Tensor) -> Tensor:
        # the forward function is a little bit modified: the central cropping as well as resizing are omitted, since the
        # images we use in our training are already cropped and resized in saving-aspect-ratio way
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img



class ViT_image_preprocessor(nn.Module):
    """The class taken from the torchvision library for image preprocesing in case of using ViT models.
    The link to GitHub is: https://github.com/pytorch/vision
    It should be noted, that the forward function has been modified:
    The central cropping as well as resizing are omitted"""
    def __init__(self) -> None:
        super().__init__()
        self.mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
        self.std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    def forward(self, img: Tensor) -> Tensor:
        # the forward function is a little bit modified: the central cropping as well as resizing are omitted, since the
        # images we use in our training are already cropped and resized in saving-aspect-ratio way
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img



def preprocess_image_MobileNetV3(image:torch.Tensor)->torch.Tensor:
    """
    Preprocesses an image or batch of images for MobileNetV3 model.
    Args:
        image: torch.Tensor
            Either a single image or a batch of images. The image should be in RGB format.

    Returns: torch.Tensor
        The preprocessed image or batch of images.

    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not isinstance(image, torch.Tensor):
        image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image, torch.float)
    image = F.normalize(image, mean=mean, std=std)
    return image


def resize_image_to_224_saving_aspect_ratio(image:torch.Tensor)-> torch.Tensor:
    # TODO: redo it using only torch
    expected_size = 224
    # transform to PIL image
    im = image.permute(1,2,0).cpu().detach().numpy()
    im = Image.fromarray(im)

    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(expected_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # create a new image and paste the resized on it
    im = im.resize(new_size, Resampling.BILINEAR)


    new_im = Image.new("RGB", (expected_size, expected_size))
    new_im.paste(im, ((expected_size - new_size[0]) // 2,
                      (expected_size - new_size[1]) // 2))
    # transform back to torch.Tensor
    new_im = F.pil_to_tensor(new_im)
    return new_im


def resize_image_saving_aspect_ratio(image:Union[torch.Tensor, np.ndarray], expected_size:int)-> \
        Union[torch.Tensor, np.ndarray]:
    # transform to PIL image if the image is torch.Tensor
    if isinstance(image, torch.Tensor):
        im = image.permute(1,2,0).cpu().detach().numpy()
        im = Image.fromarray(im)
    else:
        im = Image.fromarray(image)

    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(expected_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # create a new image and paste the resized on it
    im = im.resize(new_size, Resampling.BILINEAR)


    new_im = Image.new("RGB", (expected_size, expected_size))
    new_im.paste(im, ((expected_size - new_size[0]) // 2,
                      (expected_size - new_size[1]) // 2))
    # transform back to torch.Tensor if it was passed as torch.Tensor
    if isinstance(image, torch.Tensor):
        new_im = F.pil_to_tensor(new_im)
    else:
        # or to np.ndarray if it was passed as np.ndarray
        new_im = np.array(new_im)
    return new_im

