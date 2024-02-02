from typing import List, Union, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace
from deepface.basemodels import VGGFace, Facenet, OpenFace, ArcFace, Dlib, SFace
from deepface.basemodels.Dlib import DlibResNet


def load_deepface_model(model_name:str):
    available_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace", "Dlib", "SFace"]
    if model_name not in available_models:
        raise AttributeError(f"Unknown model name. Available models are: {available_models}")
    if model_name == "VGG-Face":
        model = VGGFace.loadModel()
    elif model_name == "Facenet":
        model = Facenet.load_facenet128d_model()
    elif model_name == "Facenet512":
        model = Facenet.load_facenet512d_model()
    elif model_name == "OpenFace":
        model = OpenFace.load_model()
    elif model_name == "ArcFace":
        model = ArcFace.load_model()
    elif model_name == "Dlib":
        model = DlibResNet()
    elif model_name == "SFace":
        model = SFace.load_model()
    return model



def recognize_faces(img:Union[str, np.ndarray], detector:str='retinaface')->List[np.ndarray]:
    """ Recognizes faces in the provided image. Returns list of bboxes for each face.

    :param img: Union[str, np.ndarray]
        The image to recognize faces in. Can be either path to image or numpy array.
    :param detector: str
        The detector to use. Can be 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', or 'fastmtcnn'.
    :return: List[np.ndarray]
        List of bboxes for each face in the image.
    """
    if detector not in ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', 'fastmtcnn']:
        raise AttributeError(f"Unknown detector. Available detectors are: ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', 'fastmtcnn']")
    bboxes = DeepFace.detection.extract_faces(img_path=img, detector_backend=detector, enforce_detection=False)
    # format of bboxes is List[Dict[str, Any]], change to List[np.ndarray]
    bboxes = [np.array([bbox['facial_area']['x'], bbox['facial_area']['y'],
                        bbox['facial_area']['x']+bbox['facial_area']['w'],
                        bbox['facial_area']['y']+bbox['facial_area']['h']])
              for bbox in bboxes]
    if len(bboxes)==1:
        bbox = bboxes[0]
        if bbox[0]==0 and bbox[1]==0 and bbox[2]==img.shape[1] and bbox[3]==img.shape[0]:
            return None
    return bboxes


def verify_two_images(img1:Union[str, np.ndarray], img2:Union[str, np.ndarray], model_name:str,
                      return_distance:Optional[bool]=False, enforce_detection:Optional[bool]=False)->Union[Tuple[bool, float], bool]:
    """ Compares two images and returns the similarity score.

    :param img1: Union[str, np.ndarray]
        The first image to compare. Can be either path to image or numpy array.
    :param img2: Union[str, np.ndarray]
        The second image to compare. Can be either path to image or numpy array.
    :param model_name: str
        The model to use for comparison. Can be 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'ArcFace', 'Dlib', or 'SFace'.
    :param return_distance: Optional[bool]
        Whether to return the distance between the two images. If True, returns tuple (verified, distance).
    :param enforce_detection: Optional[bool]
        Whether to enforce face detection. If True, throws exception if no face is detected. Set it false if the face has been detected before.
    :return: float
        The similarity score between the two images.
    """
    verification = DeepFace.verify(img1_path=img1, img2_path=img2,
                                   model_name=model_name, detector_backend='retinaface',
                                   enforce_detection=enforce_detection)
    # return distance
    if return_distance:
        return (verification['verified'], verification['distance'])
    return verification['verified']

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






if __name__=="__main__":
    # generate random data and run training of simple CNN in tensorflow2

    img1 = "/work/home/dsu/Datasets/1.jpg"
    img2 = "/work/home/dsu/Datasets/2.jpg"
    img1 = np.array(Image.open(img1))
    img2 = np.array(Image.open(img2))

    a=DeepFace.verify(img1_path=img1, img2_path=img2, model_name="ArcFace")
    print(a)

    face = DeepFace.detection.extract_faces(img_path=img2, detector_backend="retinaface")
    print(face)

    """ x = tf.random.uniform(shape=(1000, 224, 224, 3))
        y = tf.random.uniform(shape=(1000, 1))
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x, y, epochs=3, batch_size=16)"""