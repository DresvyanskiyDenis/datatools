import torch


def convert_image_to_float_and_scale(image:torch.Tensor)->torch.Tensor:
    """ Converts image to float and scales it to range [0,1]

    :param image: torch.Tensor
            image to convert
    :return: torch.Tensor
            converted image
    """
    image = image.float() / 255.
    return image