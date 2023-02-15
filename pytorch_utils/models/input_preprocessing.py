from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

class EfficientNet_image_preprocessor(nn.Module):
    """The class taken from the torchvision library for image preprocesing in case of using EfficientNet models.
    The link to GitHub is: https://github.com/pytorch/vision
    It should be noted, that the forward function has been modified:
    The central cropping as well as resizing are omitted"""
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        # the forward function is a little bit modified: the central cropping as well as resizing are omitted, since the
        # images we use in our training are already cropped and resized in saving-aspect-ratio way
        #self.resize_size = self.crop_size
        #img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        #img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )



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


def resize_image_saving_aspect_ratio(image:torch.Tensor, expected_size:int)-> torch.Tensor:
    # TODO: redo it using only torch
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

