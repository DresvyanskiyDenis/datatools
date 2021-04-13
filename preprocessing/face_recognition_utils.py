import cv2
import insightface

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == '__main__':
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=-1, nms=0.4)
    im = cv2.imread(r"C:\Users\Denis\Pictures\Camera Roll\WIN_20210413_21_38_48_Pro.jpg")[:, :, ::-1]
    im = cv2.resize(im, (640, 480))
    bbox, landmark = model.detect(im, threshold=0.5, scale=1.0)

    draw_faces(im, bbox[:,:4])
    cv2.imshow('hihi', im[:, :, ::-1])
    cv2.waitKey()

