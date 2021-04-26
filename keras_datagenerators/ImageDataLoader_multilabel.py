#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""
import multiprocessing
from itertools import repeat
from typing import Optional, Callable, Tuple, List

from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np

from preprocessing.data_preprocessing.image_preprocessing_utils import shear_image, rotate_image, flip_image, \
    shift_image, change_brightness, zoom_image, crop_image, channel_random_noise, blur_image, \
    get_image_with_worse_quality, load_image


class ImageDataLoader_multilabel(Sequence):
    """TODO:write description"""
    horizontal_flip: float
    vertical_flip: float
    horizontal: float
    brightness: float
    shearing: float
    zooming: float
    random_cropping_out: float
    bluring: float
    rotation: float
    scaling: float
    channel_random_noise: float
    worse_quality: float
    mixup: float
    preprocess_function: Callable

    prob_factors_for_each_class: Optional[Tuple[float, ...]]
    paths_with_labels: pd.DataFrame
    class_columns: List[str]
    batch_size: int
    num_classes: int
    num_workers: int
    pool: multiprocessing.Pool

    def __init__(self, paths_with_labels: pd.DataFrame, batch_size: int, class_columns: List[str],
                 preprocess_function: Optional[Callable] = None,
                 num_classes: Optional[int] = None,
                 horizontal_flip: Optional[float] = None, vertical_flip: Optional[float] = None,
                 shift: Optional[float] = None,
                 brightness: Optional[float] = None, shearing: Optional[float] = None, zooming: Optional[float] = None,
                 random_cropping_out: Optional[float] = None, rotation: Optional[float] = None,
                 scaling: Optional[float] = None,
                 channel_random_noise: Optional[float] = None, bluring: Optional[float] = None,
                 worse_quality: Optional[float] = None,
                 mixup: Optional[float] = None,
                 prob_factors_for_each_class: Optional[Tuple[float, ...]] = None,
                 pool_workers: int = 4):
        # TODO: write description
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shift = shift
        self.brightness = brightness
        self.shearing = shearing
        self.zooming = zooming
        self.random_cropping_out = random_cropping_out
        self.rotation = rotation
        self.scaling = scaling
        self.channel_random_noise = channel_random_noise
        self.bluring = bluring
        self.worse_quality = worse_quality
        self.mixup = mixup
        self.prob_factors_for_each_class = prob_factors_for_each_class

        self.paths_with_labels = paths_with_labels
        self.class_columns = class_columns
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.preprocess_function = preprocess_function
        self.num_workers = pool_workers
        # check provided params
        self._check_provided_params()
        # shuffle before start
        self.on_epoch_end()

    def _check_provided_params(self):
        # TODO: write description
        # checking the provided DataFrame
        if not set(self.class_columns).issubset(self.paths_with_labels.columns.to_list()):
            raise AttributeError('Dataframe does not contain provided class_columns. '
                                 'Dataframe columns:%s, Got %s.'
                                 % (self.paths_with_labels.columns.to_list(), self.class_columns))
        if self.paths_with_labels.shape[0] == 0:
            raise AttributeError('DataFrame is empty.')
        # check if all provided variables are in the allowed range (usually, from 0..1 or bool)
        if self.horizontal_flip is not None and (self.horizontal_flip < 0 or self.horizontal_flip > 1):
            raise AttributeError('Parameter horizontal_flip should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.vertical_flip is not None and (self.vertical_flip < 0 or self.vertical_flip > 1):
            raise AttributeError('Parameter vertical_flip should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.shift is not None and (self.shift < 0 or self.shift > 1):
            raise AttributeError('Parameter shift should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.brightness is not None and (self.brightness < 0 or self.brightness > 1):
            raise AttributeError('Parameter brightness should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.shearing is not None and (self.shearing < 0 or self.shearing > 1):
            raise AttributeError('Parameter shearing should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.zooming is not None and (self.zooming < 0 or self.zooming > 1):
            raise AttributeError('Parameter zooming should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.random_cropping_out is not None and (self.random_cropping_out < 0 or self.random_cropping_out > 1):
            raise AttributeError('Parameter random_cropping_out should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.rotation is not None and (self.rotation < 0 or self.rotation > 1):
            raise AttributeError('Parameter rotation should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.channel_random_noise is not None and (self.channel_random_noise < 0 or self.channel_random_noise > 1):
            raise AttributeError('Parameter channel_random_noise should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.bluring is not None and (self.bluring < 0 or self.bluring > 1):
            raise AttributeError('Parameter bluring should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.worse_quality is not None and (self.worse_quality < 0 or self.worse_quality > 1):
            raise AttributeError('Parameter worse_quality should be float number between 0 and 1, '
                                 'representing the probability of applying such augmentation technique.')
        if self.mixup is not None and (self.mixup < 0 or self.mixup > 1):
            raise AttributeError('Parameter mixup should be float number between 0 and 1, '
                                 'representing the portion of images to be mixup applied.')
        # create a pool of workers to do multiprocessing during loading and preprocessing
        self.pool = multiprocessing.Pool(self.num_workers)
        # calculate the number of classes if it is not provided
        if self.num_classes is None:
            self.num_classes = self.paths_with_labels.iloc[:,1].unique().shape[0]
        # check if provided len of prob_factors_for_each_class is the same as num_classes
        if self.prob_factors_for_each_class is not None:
            if len(self.prob_factors_for_each_class) != self.num_classes:
                raise AttributeError('prob_factors_for_each_class should have num_classes elements. Got %i.' % len(
                    self.prob_factors_for_each_class))
        else:
            # assign every factor to 1
            self.prob_factors_for_each_class = tuple(1. for _ in range(self.num_classes))

    def _load_and_preprocess_batch(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: write description
        filenames = self.paths_with_labels['filename'].iloc[
                    idx * self.batch_size:(idx + 1) * self.batch_size].values.flatten()
        labels = self.paths_with_labels[self.class_columns].iloc[
                 idx * self.batch_size:(idx + 1) * self.batch_size].values
        results = []
        for filename_idx in range(filenames.shape[0]):
            results.append(self.pool.apply_async(ImageAugmentor.load_and_preprocess_one_image,
                                                 args=(filenames[filename_idx],
                                                       self.horizontal_flip,
                                                       self.vertical_flip,
                                                       self.shift,
                                                       self.brightness,
                                                       self.shearing,
                                                       self.zooming,
                                                       self.random_cropping_out,
                                                       self.rotation,
                                                       self.channel_random_noise,
                                                       self.bluring,
                                                       self.worse_quality)
                                                 ))
        result = []
        for res in results:
            result.append(res.get())
        result = dict(result)
        # create batch output
        image_shape = result[filenames[0]].shape
        result_data = np.zeros((labels.shape[0],) + image_shape)
        result_labels = np.zeros(labels.shape + (self.num_classes,))
        for idx_filename in range(filenames.shape[0]):
            filename = filenames[idx_filename]
            result_data[idx_filename] = result[filename]
        # one-hot-label encoding
        for label_type_idx in range(len(self.class_columns)):
            result_labels[:, label_type_idx] = np.eye(self.num_classes)[
                labels[:, label_type_idx].reshape((-1,)).astype('int32')]
        # mixup
        if self.mixup is not None:
            result_data, result_labels = self._mixup(result_data, result_labels)

        return (result_data.astype('float32'), result_labels)

    def _mixup_data(self, data:np.ndarray, beta_values)-> np.ndarray:
        # TODO: write description
        if data.shape[0]%2!=0:
            raise AttributeError('The number of provided data instances during mixup should be odd. Got %i.'%(data.shape[0]))
        if data.shape[0]/2!=beta_values.shape[0]:
            raise AttributeError('The number of images should be the double value of number of beta values. '
                                 'Got images:%i, beta_values:%i.'%(data.shape[0], beta_values.shape[0]))
        middle_point=data.shape[0]//2
        data_left_part=data[:middle_point]
        data_right_part=data[middle_point:]
        # to make broadcasting possible
        beta_values_shape=[1 for i in range(len(data.shape))]
        beta_values_shape[0]=-1
        beta_values=beta_values.reshape(tuple(beta_values_shape))
        # generation of new mixup images
        # first part is data=first_data_instance*beta+second_data_instance*(1-beta)
        new_generated_data_part_1=data_left_part*beta_values+data_right_part*(1.-beta_values)
        # second part is data=first_data_instance*(1-beta)+second_data_instance*beta
        new_generated_data_part_2=data_left_part*(1.-beta_values)+data_right_part*beta_values
        # concatenate generated parts
        new_generated_data=np.concatenate([new_generated_data_part_1, new_generated_data_part_2])
        return new_generated_data


    def _mixup_multi_labels(self, labels:np.ndarray, beta_values:np.ndarray)->np.ndarray:
        # TODO: write description
        # labels has shape (num_instances, num_labels_type, num_classes)
        for label_type_idx in range(labels.shape[1]):
            labels[:,label_type_idx]=self._mixup_data(labels[:, label_type_idx], beta_values)
        return labels

    def _mixup(self, images: np.ndarray, labels: np.ndarray, alfa: float = 0.2)->Tuple[np.ndarray, np.ndarray]:
        # TODO: write description
        # TODO: finish this function
        portion = int(np.ceil(self.mixup * images.shape[0]))
        if portion % 2 == 1: portion += 1
        if portion == 0: return images, labels
        # generate permutations for choosing portion number of instances for mixup
        indexes_to_choose = np.random.permutation(images.shape[0])
        # generate beta values
        beta_values = np.random.beta(alfa, alfa, portion // 2)
        # save indexes of mixup and non-mixup data and labels
        indexes_for_mixup=indexes_to_choose[:portion]
        indexes_without_mixup = indexes_to_choose[portion:]
        # generate mixup data and labels
        mixup_data=self._mixup_data(images[indexes_for_mixup], beta_values)
        mixup_labels=self._mixup_multi_labels(labels[indexes_for_mixup], beta_values)
        # take non_mixup data and labels
        non_mixup_data=images[indexes_without_mixup]
        non_mixup_labels=labels[indexes_without_mixup]
        # concatenate them and shuffle
        result_data=np.concatenate([mixup_data, non_mixup_data], axis=0)
        result_labels=np.concatenate([mixup_labels, non_mixup_labels], axis=0)
        permutations=np.random.permutation(result_data.shape[0])
        result_data, result_labels = result_data[permutations], result_labels[permutations]
        return result_data, result_labels

    def on_epoch_end(self):
        # TODO: write description
        # just shuffle rows in dataframe
        self.paths_with_labels = self.paths_with_labels.sample(frac=1)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: write description
        data, labels = self._load_and_preprocess_batch(index)
        if self.preprocess_function is not None:
            data = self.preprocess_function(data)
        return (data, labels)

    def __len__(self) -> int:
        # TODO: write description
        num_steps = int(np.ceil(self.paths_with_labels.shape[0] / self.batch_size))
        return num_steps


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
