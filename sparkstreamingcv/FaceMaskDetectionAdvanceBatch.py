from pyspark.sql import SparkSession
import os
import cv2 as cv
import numpy as np
from pyspark.sql import functions as f
from pyspark.sql.types import *
import mlflow
import tensorflow as tf


# deseriallize byte to opencv image format
def convert_byte_to_nparr(img_byte):
    np_array = cv.imdecode(np.asarray(bytearray(img_byte)), cv.IMREAD_COLOR)
    return np_array


# serialize opencv image format to byte
def convert_nparr_to_byte(img_np_array):
    success, img = cv.imencode('.png', img_np_array)
    return img.tobytes()


# save image byte to s3
def write_img_byte_to_s3(s3_client, img_byte, bucket_name, img_path):
    # set the path of where you want to put the object
    img_object = s3_client.Object(bucket_name, img_path)
    # set the content which you want to write
    img_object.put(Body=img_byte)


# column function for extract image name
def extract_file_name(path):
    return f.substring_index(path, "/", -1)


# set mlflow env var
def set_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = 'https://mlflow.lab.sspcloud.fr/'
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
    os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "TdsqEs8FxPecHyixK6gI"
    if "AWS_SESSION_TOKEN" in os.environ:
        del os.environ["AWS_SESSION_TOKEN"]


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
    # fetch local model
    vgg19_model = tf.keras.models.load_model("{}/{}".format(vgg19_model_path, vgg19_model_name))
    # fetch model from mlflow server
    # set mlflow env
    set_mlflow_env()
    # Get model
    # model_name = "face-mask-detection"
    # stage = 'Production'
    # vgg19_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{stage}")
    score = vgg19_model.predict(img)
    if np.argmax(score) == 0:
        res = True
    else:
        res = False
    # print(res)
    return res


Face_Mask_Prediction_UDF = f.udf(lambda face_image_content: face_mask_prediction(face_image_content), BooleanType())


########################################### Step 3 ######################################################
def get_face_coordinate_of_origin_image(face_image_name):
    x = face_image_name.split("_")[1][1:]
    y = face_image_name.split("_")[2][1:]
    w = face_image_name.split("_")[3][1:]
    h = face_image_name.split("_")[4][1:].split('.')[0]
    return int(x), int(y), int(w), int(h)


def integrate_face_mask_prediction(origin_image_name, face_list, origin_image_content):
    final_output_path = "/tmp/sparkcv/output/final/{}".format(origin_image_name)
    buffer_img = cv.imdecode(np.asarray(bytearray(origin_image_content)), cv.IMREAD_COLOR)
    for face in face_list:
        face_image_name = face[0]
        has_mask = face[1]
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
        buffer_img = cv.putText(buffer_img, mask_label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                mask_label_color[mask_label], 2)
        # Insert a rectangle around the face
        buffer_img = cv.rectangle(buffer_img, (x, y), (x + w, y + h), mask_label_color[mask_label], 1)
    # Save the image
    cv.imwrite(final_output_path, buffer_img)
    # serialize cv image to bytes
    img_bytes = convert_nparr_to_byte(buffer_img)
    return img_bytes


Integrate_Face_Mask_Prediction_UDF = f.udf(
    lambda origin_img_name, face_list, origin_img_content: integrate_face_mask_prediction(origin_img_name, face_list,
                                                                                          origin_img_content),
    BinaryType())


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
        .load(image_input_folder_path) \
        .withColumn("time_stamp", f.current_timestamp()) \
        # imgbyte = list(raw_images_df.select("content").toPandas()["content"])[0]
    # # print(imgbyte)
    # img = cv2.imdecode(np.asarray(bytearray(imgbyte)), cv2.IMREAD_COLOR)
    # print(img)
    image_name_df = raw_image_df \
        .select("path", "content", "time_stamp") \
        .withColumn("origin_image_name", extract_file_name(f.col("path"))).drop("path")
    image_name_df.show()

    detected_face_list_df = image_name_df.withColumn("detected_face_list",
                                                     Face_Extraction_UDF("origin_image_name", "content"))

    detected_face_ob_df = detected_face_list_df.withColumn("extracted_face",
                                                           f.explode(f.col("detected_face_list"))).drop(
        "detected_face_list")
    detected_face_ob_df.printSchema()
    detected_face_df = detected_face_ob_df.select(f.col("origin_image_name"), f.col("time_stamp"), f.col("content"),
                                                  f.col("extracted_face.img_name").alias("extracted_face_image_name"),
                                                  f.col("extracted_face.img_content").alias(
                                                      "extracted_face_image_content"))
    detected_face_df.show()
    # step 2
    predicted_mask_df = detected_face_df.withColumn("with_mask",
                                                    Face_Mask_Prediction_UDF("extracted_face_image_content"))
    predicted_mask_df.show()

    # step 3
    grouped_face_df = predicted_mask_df.drop("extracted_face_image_content").groupBy("origin_image_name",
                                                                                     "content").agg(
        f.collect_list(
            f.struct(
                *[f.col("extracted_face_image_name").alias("face_name"), f.col("with_mask").alias("with_mask")]))
            .alias("face_list"))
    grouped_face_df.show()
    # join_with_content_df = grouped_face_df.join(image_name_df, "origin_image_name", "inner")
    # join_with_content_df.show()
    final_df = grouped_face_df.withColumn("marked_img_content",
                                          Integrate_Face_Mask_Prediction_UDF("origin_image_name", "face_list",
                                                                             "content"))
    final_df.show()


if __name__ == "__main__":
    main()
