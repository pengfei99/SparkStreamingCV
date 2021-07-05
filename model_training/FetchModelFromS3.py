import cv2 as cv
import numpy as np
import tensorflow as tf


def main():
    # fetch model via https
    _URL = "https://minio.lab.sspcloud.fr/pengfei/diffusion/computer_vision/trained_model/masknet.h5"
    model_path = tf.keras.utils.get_file('masknet.h5', origin=_URL)
    vgg19_model = tf.keras.models.load_model(model_path)

    # prepare sample data
    path = "/tmp/sparkcv/output/faces/hasMask1.png"
    img = cv.imread(path)
    img = cv.resize(img, (128, 128))
    img = np.reshape(img, [1, 128, 128, 3])
    img = img / 255.0

    # give a prediction
    score = vgg19_model.predict(img)
    if np.argmax(score) == 0:
        res = True
    else:
        res = False
    print(res)


if __name__ == "__main__":
    main()
