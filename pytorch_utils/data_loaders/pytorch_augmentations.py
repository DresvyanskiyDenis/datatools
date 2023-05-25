from typing import Union, Tuple

import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms as T


def pad_image_random_factor(image:torch.Tensor)->torch.Tensor:
    old_shape = image.shape
    pad_factor=np.random.randint(1,50)
    image = T.Pad(padding=pad_factor)(image)
    image = T.RandomCrop(old_shape[1:])(image)
    return image

def grayscale_image(image:torch.Tensor)->torch.Tensor:
    return T.Grayscale(num_output_channels=3)(image)

def collor_jitter_image_random(image:torch.Tensor, brightness:float, hue:float, contrast:float, saturation:float)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
    return T.ColorJitter(brightness=brightness, hue=hue, contrast=contrast, saturation=saturation)(image)

def gaussian_blur_image_random(image:torch.Tensor, kernel_size:Tuple[int, int], sigma:Tuple[float, float])->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html#torchvision.transforms.GaussianBlur
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(image)

def random_perspective_image(image:torch.Tensor)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomPerspective.html#torchvision.transforms.RandomPerspective
    distortion_scale = (0.3-0.8)*torch.rand(1) + 0.8
    return T.RandomPerspective(distortion_scale=distortion_scale, p=1.)(image)

def random_rotation_image(image:torch.Tensor)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomRotation.html#torchvision.transforms.RandomRotation
    angle = torch.randint(0,90,size=(1,)).item()
    return T.RandomRotation(degrees=angle)(image)

def random_crop_image(image:torch.Tensor, cropping_factor_limits:Tuple[float, float])->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop
    old_shape = image.shape
    crop_factor = (cropping_factor_limits[0]- cropping_factor_limits[1])* torch.rand(1) + cropping_factor_limits[1]
    crop_shape = (int(old_shape[1]*crop_factor), int(old_shape[2]*crop_factor))
    image = T.RandomCrop(crop_shape)(image)
    image = T.Resize(old_shape[1:], antialias=True)(image)
    return image

def random_posterize_image(image:torch.Tensor)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomPosterize.html#torchvision.transforms.RandomPosterize
    bits = torch.randint(1,3,size=(1,)).item()
    return T.RandomPosterize(bits=bits)(image)

def random_adjust_sharpness_image(image:torch.Tensor, sharpness_factor_limits:Tuple[float, float])->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAdjustSharpness.html#torchvision.transforms.RandomAdjustSharpness
    sharpness_factor = (sharpness_factor_limits[0]- sharpness_factor_limits[1])* torch.rand(1) + sharpness_factor_limits[1]
    return T.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=1.)(image)

def random_equalize_image(image:torch.Tensor)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomEqualize.html#torchvision.transforms.RandomEqualize

    try:
        return T.RandomEqualize(p=1.)(image)
    except:
        return image

def random_horizontal_flip_image(image:torch.Tensor)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip
    return T.RandomHorizontalFlip(p=1.)(image)

def random_vertical_flip_image(image:torch.Tensor)->torch.Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomVerticalFlip.
    return T.RandomVerticalFlip(p=1.)(image)




