from pyspark.sql import SparkSession
import os
import boto3
from botocore.client import Config
import cv2 as cv
import numpy as np
from pyspark.sql import functions as f
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf


def convert_byte_to_nparr(img_byte):
    np_array = cv.imdecode(np.asarray(bytearray(img_byte)), cv.IMREAD_COLOR)
    return np_array


def convert_nparr_to_byte(img_np_array):
    success, img = cv.imencode('.png', img_np_array)
    return img.tobytes()


def write_img_to_S3(s3_client, img_np_array, bucket_name, img_path):
    success, my_img = cv.imencode('.png', img_np_array)
    my_img_byte = my_img.tobytes()
    # set the path of where you want to put the object
    img_object = s3_client.Object(bucket_name, img_path)
    # set the content which you want to write
    img_object.put(Body=my_img_byte)


def extract_file_name(path):
    return f.substring_index(path, "/", -1)


def face_extraction(image_name, raw_img_content):
    haar_model_name = "haarcascade_frontalface_default.xml"
    haar_model_path = "/mnt/hgfs/Centos7_share_folder/trained_models/"
    bucket_name = "pengfei"
    faces_output_path = "tmp/sparkcv/output/faces/"
    path = "{}{}".format(haar_model_path, haar_model_name)
    img = cv.imdecode(np.asarray(bytearray(raw_img_content)), cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.IMREAD_GRAYSCALE)
    face_model = cv.CascadeClassifier(path)
    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)  # returns a list of (x,y,w,h) tuples

    # Extract faces from the origin image
    extracted_face_list = []
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        img_content = img[y:y + h, x:x + w]
        img_content = cv.resize(img_content, (128, 128))
        extracted_face_img_name = image_name[:-4] + "_x" + str(x) + "_y" + str(y) + "_w" + str(
            w) + "_h" + str(h) + ".png"
        img_byte = convert_nparr_to_byte(img_content)

        extracted_face_list.append((extracted_face_img_name, img_byte))
    # extracted_face_output_path = faces_output_path + extracted_face_img_name
    # write_img_to_S3(s3_client, img_content, bucket_name, extracted_face_output_path)
    return extracted_face_list


face_extraction_schema = ArrayType(StructType([
    StructField("img_name", StringType(), False),
    StructField("img_content", BinaryType(), False)
]))

Face_Extraction_UDF = f.udf(lambda image_name, raw_image_content: face_extraction(image_name, raw_image_content),
                            face_extraction_schema)


################################################## Step2 ################################################
def face_mask_prediction(np_img_str):
    vgg19_model_path = "/mnt/hgfs/Centos7_share_folder/trained_models"
    vgg19_model_name = "masknet.h5"
    # read raw face image
    np_arr_img = convert_byte_to_nparr(np_img_str)
    img = np.reshape(np_arr_img, [1, 128, 128, 3])
    img = img / 255.0
    vgg19_model = tf.keras.models.load_model("{}/{}".format(vgg19_model_path, vgg19_model_name))
    score = vgg19_model.predict(img)
    if np.argmax(score) == 0:
        res = True
    else:
        res = False
    # print(res)
    return res


Face_Mask_Prediction_UDF = f.udf(lambda face_image_content: face_mask_prediction(face_image_content), BooleanType())


def main():
    os.environ['SPARK_HOME'] = "/home/pliu/Tools/spark/spark-3.1.2"

    image_input_folder_path = "/tmp/sparkcv/input/"
    # image_input_folder_path = "s3a://{}/tmp/sparkcv/input/toto.png".format(bucket_name)
    spark = SparkSession.builder \
        .master("local") \
        .appName("FaceDetection") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0") \
        .getOrCreate()

    image_schema = spark.read.format("binaryFile").load(image_input_folder_path).schema
    raw_image_df = spark.read \
        .format("binaryFile") \
        .schema(image_schema) \
        .option("maxFilesPerTrigger", "500") \
        .option("recursiveFileLookup", "true") \
        .option("pathGlobFilter", "*.png") \
        .load(image_input_folder_path)
    # imgbyte = list(raw_images_df.select("content").toPandas()["content"])[0]
    # # print(imgbyte)
    # img = cv2.imdecode(np.asarray(bytearray(imgbyte)), cv2.IMREAD_COLOR)
    # print(img)
    image_name_df = raw_image_df \
        .select("path", "content") \
        .withColumn("origin_image_name", extract_file_name(f.col("path"))).drop("path")
    image_name_df.show()
    detected_face_list_df = image_name_df.withColumn("detected_face_list",
                                                     Face_Extraction_UDF("origin_image_name", "content")).drop(
        "content")
    detected_face_ob_df = detected_face_list_df.withColumn("extracted_face",
                                                           f.explode(f.col("detected_face_list"))).drop(
        "detected_face_list")
    detected_face_ob_df.printSchema()
    detected_face_df = detected_face_ob_df.select(f.col("origin_image_name"),
                                                  f.col("extracted_face.img_name").alias("extracted_face_image_name"),
                                                  f.col("extracted_face.img_content").alias(
                                                      "extracted_face_image_content"))
    detected_face_df.show()
    # step 2
    with_mask_df = detected_face_df.withColumn("with_mask", Face_Mask_Prediction_UDF("extracted_face_image_content"))
    with_mask_df.show()


if __name__ == "__main__":
    main()
