"""
The RetinaFace model is taken from GutHub repositories https://github.com/adas-eye/retinaface
and https://github.com/biubug6/Pytorch_Retinaface (main repository)
Thanks to the authors for their work. For more details, please refer to the original repository.
Do not forget to cite https://github.com/biubug6/Pytorch_Retinaface if you use RetinaFace in your work.
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2022"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from typing import List, Union

import torch
from PIL import Image
import numpy as np
from feature_extraction.pytorch_based.retinaface import RetinafaceDetector

def load_and_prepare_detector_retinaFace_mobileNet(device:str):
    """
    Constructs and initializes RetinaFace model with Mobinet backbone
    :return: RetinaFace model
    """
    model = RetinafaceDetector(net='mnet', type=device).detect_faces
    return model

def get_most_confident_person(recognized_faces:List[List[float]])->List[float]:
    """
    Finds the most confident face in the list of recognized faces
    :param recognized_faces: List[List[float]]
            List of faces, which were recognized by RetinaFace model
    :return: List[float]
            List of 5 floats, which represent the bounding box of face and its confidence
    """
    return max(recognized_faces, key=lambda x: x[4])

def recognize_one_face_bbox(img:Union[np.ndarray, str], detector:RetinafaceDetector.detect_faces)->List[float]:
    """
    Recognizes the faces in provided image and return the face with the highest confidence.
    :param img: np.ndarray
            image represented by np.ndarray or path to the image
    :param detector: object
            the model, which has method detect. It should return bounding boxes and landmarks.
    :param threshold: float
            adjustable parameter for recognizing if detected object is face or not.
    :return: List[float]
            List of 5 floats, which represent the bounding box of face and its confidence
    """
    if type(img) is str:
        img = np.array(Image.open(img))
    recognized_faces = detector(img)
    if recognized_faces is None or len(recognized_faces) == 0:
        return None
    return get_most_confident_person(recognized_faces)

def extract_face_according_bbox(img:np.ndarray, bbox:List[float])->np.ndarray:
    """
    Extracts (crops) image according provided bounding box to get only face.
    :param img: np.ndarray
            image represented by np.ndarray
    :param bbox: List[float]
            List of 5 floats, which represent the bounding box of face and its confidence
    :return: np.ndarray
            image represented by np.ndarray
    """
    x1, y1, x2, y2 = bbox[:4]
    # take a little more than the bounding box
    x1,y1,x2,y2 = int(x1-10), int(y1-10), int(x2+10), int(y2+10)
    # check if the bounding box is out of the image
    x1 = max(x1,0)
    y1 = max(y1,0)
    x2 = min(x2,img.shape[1])
    y2 = min(y2,img.shape[0])
    return img[y1:y2, x1:x2]


def recognize_one_face(img:Union[np.ndarray, str], detector:RetinafaceDetector.detect_faces)->np.ndarray:
    """
    Recognizes the faces in provided image and return the face with the highest confidence.
    :param img: np.ndarray
            image represented by np.ndarray or path to the image
    :param detector: object
            the model, which has method detect. It should return bounding boxes and landmarks.
    :param threshold: float
            adjustable parameter for recognizing if detected object is face or not.
    :return: np.ndarray
            image represented by np.ndarray
    """
    if type(img) is str:
        img = np.array(Image.open(img))
    bbox = recognize_one_face_bbox(img, detector)
    if bbox is None:
        return None
    return extract_face_according_bbox(img, bbox)



if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = RetinafaceDetector(net='mnet', type=device)
    img = np.array(Image.open("/work/home/dsu/Datasets/1.jpg"))
    detector.detect_faces(img)


    """from torchsummary import summary
    summary(detector, (3, 224, 224))"""