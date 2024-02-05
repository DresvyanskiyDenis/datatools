__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import Union, Tuple, List

import numpy as np


def crop_frame_by_bbox(frame: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Cuts frame according to provided bounding box.
    :param frame: np.ndarray
            image represented by np.ndarray
    :param bbox: List[int]
            List of 4 ints, which represent the bounding box of face
    :return: np.ndarray
            cropped image represented by np.ndarray
    """
    return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def get_pose_bbox(frame: np.ndarray, detector: object) -> Union[List[int], None]:
    """
    Recognizes the pose in provided image and return the bbox for pose.
    :param frame: np.ndarray
            image represented by np.ndarray
    :param detector: object
            the model, which has method predict. It should return bounding boxes.
    :return: List[int]
            List of 4 ints, which represent the bounding box of pose
    """
    prediction = detector.predict(frame)
    if prediction is None or len(prediction[0]) == 0:
        return None
    bbox = prediction[0][0]
    return bbox


def crop_frame_to_pose(frame: np.ndarray, bbox: List[int], return_bbox: bool = False, limits:List[float]=[125, 100]) -> \
        Union[Tuple[np.ndarray, List[int]], np.ndarray, None]:
    height, width, _ = frame.shape
    # expand bbox so that it will cover all human with some space
    # height
    bbox[1] -= limits[0]
    bbox[3] += limits[0]
    # width
    bbox[0] -= limits[1]
    bbox[2] += limits[1]
    # check if we are still in the limits of the frame
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[3] > height:
        bbox[3] = height
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[2] > width:
        bbox[2] = width
    # crop frame
    bbox = [int(round(x)) for x in bbox]
    cropped_frame = crop_frame_by_bbox(frame, bbox)
    if return_bbox:
        return cropped_frame, bbox
    return cropped_frame
