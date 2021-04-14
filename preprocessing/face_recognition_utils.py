from typing import Tuple

import cv2
import insightface
import numpy as np

def recognize_the_most_confident_person_retinaFace(im:np.ndarray, detector:object,
                                                   threshold:float=0.5)->Tuple[int,...]:
    # TODO: write description
    bboxes, landmark = detector.detect(im, threshold=threshold, scale=1.0)
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
    # TODO: write description
    model = insightface.model_zoo.get_model(model_name)
    model.prepare(ctx_id=-1, nms=0.4)
    return model

def extract_face_according_bbox(img:np.ndarray, bbox:Tuple[int,...])->np.ndarray:
    # TODO: write description
    x0, y0, x1, y1 = bbox
    return img[y0:y1,x0:x1]


def draw_faces(im, bboxes):
    # TODO: write description
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


    '''model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=-1, nms=0.4)
    im = cv2.imread(r"D:\Databases\DAiSEE\frames\1100011002\1100011002_frames\1100011002226.jpg")[:, :, ::-1]
    im=np.array(im)
    #im = cv2.resize(im, (480, 480))
    bbox, landmark = model.detect(im, threshold=0.5, scale=1.0)

    draw_faces(im, bbox[1:,:4])
    cv2.imshow('hihi', im[:, :, ::-1])
    cv2.waitKey()'''
