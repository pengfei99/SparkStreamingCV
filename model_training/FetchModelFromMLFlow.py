import mlflow
import os
import logging
import cv2 as cv
import numpy as np

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def set_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = 'https://mlflow.lab.sspcloud.fr/'
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
    os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "changeMe"
    del os.environ["AWS_SESSION_TOKEN"]


def main():

    ## prepare sample data
    path = "/tmp/sparkcv/output/faces/toto.png"
    img = cv.imread(path)
    img = cv.resize(img, (128, 128))
    img = np.reshape(img, [1, 128, 128, 3])
    img = img / 255.0
    # set mlflow env
    set_mlflow_env()
    # Get model
    model_name = "face-mask-detection"
    stage = 'Production'
    model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{stage}")
    # predict the sample data
    score = model.predict(img)
    if np.argmax(score) == 0:
        res = True
    else:
        res = False
    print(res)


if __name__ == "__main__":
    main()
