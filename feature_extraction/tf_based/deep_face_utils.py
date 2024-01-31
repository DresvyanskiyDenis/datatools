import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace

def load_deepface_model(model_name:str):
    available_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
    if model_name not in available_models:
        raise AttributeError(f"Unknown model name. Available models are: {available_models}")


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