#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains functions for face detection (recognition)

Module contains functions for face recognition, using RetinaFace detector and insightface lib.
RetinaFace: https://arxiv.org/abs/1905.00641
Insightface lib: https://insightface.ai/projects

List of functions:

    * recognize_the_most_confident_person_retinaFace - recognizes the faces on provided image and returns the most confident
        one (the face with the highest confidence)
    * load_and_prepare_detector_retinaFace - loads and returns RetinaFace model
    * extract_face_according_bbox - extracts (or crop) image according provided bounding box
    * draw_faces - display provided face
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import Tuple
import cv2
import insightface
import numpy as np

def recognize_the_most_confident_person_retinaFace(im:np.ndarray, detector:object,
                                                   threshold:float=0.5)->Tuple[int,...]:
    """Recognizes the faces in provided image and return the face with the highest confidence.

    :param im: np.ndarray
            image represented by np.ndarray
    :param detector: object
            the model, which has method detect. It should return bounding boxes and landmarks.
    :param threshold: float
            adjustable parameter for recognizing if detected object is face or not.
    :return: Tuple[int,...]
            Tuple of 4 ints, which represent the bounding box of face
    """
    bboxes, landmark = detector.detect(im, threshold=threshold, scale=1.0)
    # check if there are some found bboxes
    if len(bboxes)==0:
        return tuple()
    # find the most confident face (hopefully, it is that, which we need)
    max_conf=bboxes[0][-1]
    bbox_most_conf=bboxes[0][:-1]
    for bbox_idx in range(len(bboxes)):
        if max_conf<bboxes[bbox_idx][-1]:
            max_conf=bboxes[bbox_idx][-1]
            bbox_most_conf=bboxes[bbox_idx][:-1]
    # convert it to int
    bbox_most_conf=tuple(int(_) for _ in bbox_most_conf)
    return bbox_most_conf

def load_and_prepare_detector_retinaFace(model_name:str='retinaface_r50_v1')->object:
    """Loads and prepare model from insightface.model_zoo

    :param model_name: str
            name of the model to be loaded
    :return: object
            insightface lib model
    """
    model = insightface.model_zoo.get_model(model_name)
    model.prepare(ctx_id=-1, nms=0.4)
    return model

def extract_face_according_bbox(img:np.ndarray, bbox:Tuple[int,...])->np.ndarray:
    """Extracts (crops) image according provided bounding box to get only face.

    :param img: np.ndarray
            image represented by np.ndarray
    :param bbox: Tuple[int, ...]
            Tuple of 4 ints, which represent the bounding box of face
    :return: np.ndarray
            Cropped image
    """
    x0, y0, x1, y1 = bbox
    return img[y0:y1,x0:x1]


def draw_faces(im:np.ndarray, bboxes:Tuple[Tuple[int,...],...])->None:
    """Displays image with rectangled faces.

    :param im: np.ndarray
            image represented by np.ndarray
    :param bboxes:Tuple[Tuple[int,...],...]
            Several bounding boxes to be displayed on image.
    :return: None
    """
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == '__main__':
    model=load_and_prepare_detector_retinaFace()
    im = cv2.imread(r"D:\Databases\DAiSEE\frames\1100011002\1100011002_frames\1100011002226.jpg")[:, :, ::-1]
    im = np.array(im)
    bbox=recognize_the_most_confident_person_retinaFace(im,model)
    face=extract_face_according_bbox(im, bbox)
    cv2.imwrite('img.jpg',face[:,:,::-1])

