from typing import Optional
import numpy as np

from preprocessing.data_preprocessing.image_preprocessing_utils import shear_image, rotate_image, flip_image, \
    shift_image, change_brightness, zoom_image, crop_image, channel_random_noise, blur_image, \
    get_image_with_worse_quality, load_image


class ImageAugmentor():

    @staticmethod
    def _shear_image(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [-0.5, 0.5]
        randomly_picked_param = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        img = shear_image(img, shear_factor=randomly_picked_param)
        return img

    @staticmethod
    def _rotate_image(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [-90, 90]
        randomly_picked_param = np.random.randint(parameter_boundaries[0], parameter_boundaries[1] + 1)
        img = rotate_image(img, rotation_angle=randomly_picked_param)
        return img

    @staticmethod
    def _flip_image_vertical(img: np.ndarray):
        # TODO: write description
        img = flip_image(img, flip_type='vertical')
        return img

    @staticmethod
    def _flip_image_horizontal(img: np.ndarray):
        # TODO: write description
        img = flip_image(img, flip_type='horizontal')
        return img

    @staticmethod
    def _shift_image(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [-50, 50]
        randomly_picked_param_vertical = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        randomly_picked_param_horizontal = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        img = shift_image(img, shift_vector=(randomly_picked_param_horizontal, randomly_picked_param_vertical))
        return img

    @staticmethod
    def _change_brightness_image(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [-0.5, 0.5]
        randomly_picked_param = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        img = change_brightness(img, brightness_factor=randomly_picked_param)
        return img

    @staticmethod
    def _zoom_image(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [1.1, 1.5]
        old_shape = img.shape[:2]
        randomly_picked_param = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        # firstly zoom image (increase or decrease the size)
        img = zoom_image(img, zoom_factor=randomly_picked_param)
        # take part of image (for example, with size of old shape, if it became bigger)
        # x0 can be chosen in the range (0, current_img.shape[1]-old_img.shape[1])
        # y0 can be chosen in the range (0, current_img.shape[0]-old_img.shape[0])
        x0 = np.random.randint(0, img.shape[1] - old_shape[1])
        y0 = np.random.randint(0, img.shape[0] - old_shape[0])
        x1 = x0 + old_shape[1]
        y1 = y0 + old_shape[1]
        img = crop_image(img, bbox=(x0, y0, x1, y1))
        return img

    @staticmethod
    def _add_noise_on_one_channel(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [5, 20]
        randomly_picked_param = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        randomly_picked_channel = np.random.randint(0, 3)
        img = channel_random_noise(img, num_channel=randomly_picked_channel, std=randomly_picked_param)
        return img

    @staticmethod
    def _random_cutting_out(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        # choose which proportion of image should we cut out
        parameter_boundaries = [0.2, 0.4]
        randomly_picked_param = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1])
        # transform it to index according to shape f image
        randomly_picked_param = int(np.round(randomly_picked_param * (img.shape[0] + img.shape[1]) / 2.))
        # calculate randomly the area, which will be cut out
        x0 = np.random.randint(0, img.shape[1] - randomly_picked_param)
        x1 = x0 + randomly_picked_param
        y0 = np.random.randint(0, img.shape[0] - randomly_picked_param)
        y1 = y0 + randomly_picked_param
        # copy for getting rid of non-predictable changing the image
        img = img.copy()
        # cut out
        img[y0:y1, x0:x1] = 0
        return img

    @staticmethod
    def _blur_image(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [1, 3]
        randomly_picked_param = np.random.randint(parameter_boundaries[0], parameter_boundaries[1] + 1)
        img = blur_image(img, sigma=randomly_picked_param)
        return img

    @staticmethod
    def _worse_quality(img: np.ndarray):
        # TODO: write description
        # this is empirically chosen parameter boundaries
        parameter_boundaries = [0.2, 0.8]
        randomly_picked_param = np.random.uniform(parameter_boundaries[0], parameter_boundaries[1] + 1)
        img = get_image_with_worse_quality(img, rescaling_factor=randomly_picked_param)
        return img

    @staticmethod
    def load_and_preprocess_one_image(path: str, horizontal_flip: Optional[float] = None,
                                      vertical_flip: Optional[float] = None,
                                      shift: Optional[float] = None,
                                      brightness: Optional[float] = None, shearing: Optional[float] = None,
                                      zooming: Optional[float] = None,
                                      random_cropping_out: Optional[float] = None, rotation: Optional[float] = None,
                                      channel_random_noise: Optional[float] = None, bluring: Optional[float] = None,
                                      worse_quality: Optional[float] = None):
        img = ImageAugmentor._load_image(path)
        img = ImageAugmentor._augment_one_image(img, horizontal_flip, vertical_flip,
                                                shift, brightness, shearing, zooming, random_cropping_out, rotation,
                                                channel_random_noise, bluring,
                                                worse_quality)
        return path, img

    @staticmethod
    def _load_image(path):
        # TODO: write description
        img = load_image(path)
        return img

    @staticmethod
    def _augment_one_image(img: np.ndarray, horizontal_flip: Optional[float] = None,
                           vertical_flip: Optional[float] = None,
                           shift: Optional[float] = None,
                           brightness: Optional[float] = None, shearing: Optional[float] = None,
                           zooming: Optional[float] = None,
                           random_cropping_out: Optional[float] = None, rotation: Optional[float] = None,
                           channel_random_noise: Optional[float] = None, bluring: Optional[float] = None,
                           worse_quality: Optional[float] = None):
        # TODO: write description
        # horizontal flipping
        if not horizontal_flip is None and ImageAugmentor._get_answer_with_prob(horizontal_flip):
            img = ImageAugmentor._flip_image_horizontal(img)
        # vertical flipping
        if not vertical_flip is None and ImageAugmentor._get_answer_with_prob(vertical_flip):
            img = ImageAugmentor._flip_image_vertical(img)
        # shifting
        if not shift is None and ImageAugmentor._get_answer_with_prob(shift):
            img = ImageAugmentor._shift_image(img)
        # brightness changing
        if not brightness is None and ImageAugmentor._get_answer_with_prob(brightness):
            img = ImageAugmentor._change_brightness_image(img)
        # shearing
        if not shearing is None and ImageAugmentor._get_answer_with_prob(shearing):
            img = ImageAugmentor._shear_image(img)
        # zooming
        if not zooming is None and ImageAugmentor._get_answer_with_prob(zooming):
            img = ImageAugmentor._zoom_image(img)
        # channel random noise
        if not channel_random_noise is None and ImageAugmentor._get_answer_with_prob(channel_random_noise):
            img = ImageAugmentor._add_noise_on_one_channel(img)
        # rotation
        if not rotation is None and ImageAugmentor._get_answer_with_prob(rotation):
            img = ImageAugmentor._rotate_image(img)
        # random cropping out
        if not random_cropping_out is None and ImageAugmentor._get_answer_with_prob(random_cropping_out):
            img = ImageAugmentor._random_cutting_out(img)
        # bluring
        if not bluring is None and ImageAugmentor._get_answer_with_prob(bluring):
            img = ImageAugmentor._blur_image(img)
        # worse quality
        if not worse_quality is None and ImageAugmentor._get_answer_with_prob(worse_quality):
            img = ImageAugmentor._worse_quality(img)
        # return augmented image
        return img

    @staticmethod
    def _get_answer_with_prob(prob: float):
        if prob == 0:
            return False
        # TODO: write description
        if prob < 0 or prob > 1:
            raise AttributeError('Probability should be a float number between 0 and 1. Gor %s.' % prob)
        # roll the dice
        rolled_prob = np.random.uniform(0, 1)
        if rolled_prob < prob:
            return True
        return False