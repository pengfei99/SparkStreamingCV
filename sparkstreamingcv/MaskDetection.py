import pyspark.sql.functions as sql_fun
from pyspark.sql import SparkSession
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

spark = SparkSession.builder \
    .master("local") \
    .appName("FaceDetection") \
    .getOrCreate()


# Get the name of face image from the file path
def extract_face_image_name(path_col):
    return sql_fun.substring_index(path_col, "/", -1)


# Get the origin image name where the face image are extracted
def get_origin_image_name(path):
    return sql_fun.regexp_replace(path, "(_.*)\d", "")


def get_face_coordinate_of_origin_image(face_image_name):
    x = face_image_name.split("_")[1][1:]
    y = face_image_name.split("_")[2][1:]
    w = face_image_name.split("_")[3][1:]
    h = face_image_name.split("_")[4][1:].split('.')[0]
    return int(x), int(y), int(w), int(h)


# Recuperation_des_visages_UDF(col("path"), col("photo"), col("prediction")
def integrate_face_mask_prediction(face_image_name, origin_image_name, has_mask):
    image_input_folder_path = "/tmp/sparkcv/input"
    final_output_path = "/tmp/sparkcv/output/final"
    # check if the image is already treated or not, if yes, it means it has multiple faces. and we just add new mask
    # prediction to untreated faces.
    # If not treated, we load image from
    treated_image_path = "{}/{}".format(final_output_path, origin_image_name)
    print(treated_image_path)
    # If the image is treated, update the treated image
    if os.path.isfile(treated_image_path):
        image = cv2.imread(final_output_path)
    else:
        # Get the untreated image from input
        image = cv2.imread("{}/{}".format(image_input_folder_path, origin_image_name))
        print("{}/{}".format(image_input_folder_path, origin_image_name))

    # set Label text
    if has_mask:
        mask_label = "MASK"
    else:
        mask_label = "NO MASK"
    # Get the coordinate and size of face image
    (x, y, w, h) = get_face_coordinate_of_origin_image(face_image_name)

    # Set text color for mask label
    mask_label_color = {"MASK": (0, 255, 0), "NO MASK": (0, 0, 255)}

    # Insert mask label to image
    image = cv2.putText(image, mask_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        mask_label_color[mask_label], 2)
    # Insert a rectangle around the face
    image = cv2.rectangle(image, (x, y), (x + w, y + h), mask_label_color[mask_label], 1)
    # Save the image
    cv2.imwrite(treated_image_path, image)

    return "Done"


# This function use a trained vgg19 model to predict if it has mask or no. It returns true, if it has mask.
def face_mask_prediction(face_image_name):
    face_image_path = "/tmp/sparkcv/output/faces"
    vgg19_model_path = "/mnt/hgfs/Centos7_share_folder/trained_models"
    vgg19_model_name = "masknet.h5"
    # read raw face image
    img = cv2.imread("{}/{}".format(face_image_path, face_image_name))
    # normalize the raw image for vgg19 model
    img = cv2.resize(img, (128, 128))
    # plt.imshow(img)
    # plt.show()
    img = np.reshape(img, [1, 128, 128, 3])
    img = img / 255.0
    vgg19_model = tf.keras.models.load_model("{}/{}".format(vgg19_model_path, vgg19_model_name))
    score = vgg19_model.predict(img)
    if np.argmax(score) == 0:
        res = True
    else:
        res = False
    # print(res)
    return res


Face_Mask_Prediction_UDF = udf(lambda face_image_name: face_mask_prediction(face_image_name))

Integrate_Face_Mask_Prediction_UDF = udf(
    lambda face_image_name, origin_image_name, has_mask: integrate_face_mask_prediction(face_image_name,
                                                                                        origin_image_name, has_mask))


def detect_mask_in_batch(image_input_folder_path):
    schema = spark.read.format("binaryFile").load(image_input_folder_path).schema
    # read raw image as a df
    raw_image_df = spark.read \
        .format("binaryFile") \
        .schema(schema) \
        .load(image_input_folder_path)
    raw_image_df.show()
    # generate two columns face_image_name and origin_image_name
    df = raw_image_df.withColumn("extracted_face_image_name", extract_face_image_name("path")) \
        .withColumn("origin_image_name", get_origin_image_name("extracted_face_image_name")) \
        .select("path", "modificationTime", "origin_image_name", "extracted_face_image_name", "content")
    df.show()

    # generate a column by using vgg19 prediction
    predict_df = df.withColumn("has_mask", Face_Mask_Prediction_UDF("extracted_face_image_name").cast(BooleanType()))
    predict_df.show()

    # integrate the face mask prediction to origin image
    complete_df = predict_df.withColumn("integration",
                                        Integrate_Face_Mask_Prediction_UDF("extracted_face_image_name",
                                                                           "origin_image_name",
                                                                           "has_mask"))
    complete_df.show()


def main():
    face_image_input_folder_path = "/tmp/sparkcv/output/faces/"
    # face_mask_prediction("maksssksksss330_x280_y21_w73_h73.png")
    # face_mask_prediction("maksssksksss330_x283_y97_w59_h59.png")
    # integrate_face_mask_prediction("maksssksksss51_x98_y133_w126_h126.png", "maksssksksss51.png", False)
    detect_mask_in_batch(face_image_input_folder_path)


if __name__ == "__main__":
    main()
