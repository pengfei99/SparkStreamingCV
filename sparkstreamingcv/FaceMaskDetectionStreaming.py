from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf
import os

from pyspark.sql.types import StringType, ArrayType, BooleanType


def render_image(binary_content):
    img = Image.open(io.BytesIO(binary_content))
    img = img.convert('RGB')


def extract_file_name(path):
    return f.substring_index(path, "/", -1)


def face_extraction(image_name):
    image_input_folder_path = "/tmp/sparkcv/input/"
    cascade_model_path = "/mnt/hgfs/Centos7_share_folder/trained_models"
    faces_output_path = "/tmp/sparkcv/output/faces/"
    image_path = image_input_folder_path + image_name

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    # loading haarcascade_frontalface_default.xml, you can get all the pre-trained model from
    # https://github.com/opencv/opencv/tree/3.4/data/haarcascades
    face_model = cv2.CascadeClassifier("{}/haarcascade_frontalface_default.xml".format(cascade_model_path))
    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)  # returns a list of (x,y,w,h) tuples
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Extract faces from the origin image
    extracted_face_list = []
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = img[y:y + h, x:x + w]
        extracted_face_img_name = image_name[:-4] + "_x" + str(x) + "_y" + str(y) + "_w" + str(
            w) + "_h" + str(h) + ".png"
        extracted_face_list.append(extracted_face_img_name)
        extracted_face_output_path = faces_output_path + extracted_face_img_name
        cv2.imwrite(extracted_face_output_path, crop)

    return extracted_face_list


# create a udf which calls face_extraction function
# It has three part in udf*=():
# 1.lambda image_name is the input argument of the udf
# 2. face_extraction(image_name) invocation of the python function
# 3. return type of the python function, also the returned column type
Face_Extraction_UDF = f.udf(lambda image_name: face_extraction(image_name), ArrayType(StringType()))


def detect_faces_streaming(raw_image_df_stream):
    # get the image name df
    image_name_df = raw_image_df_stream \
        .select("path") \
        .withColumn("origin_image_name", extract_file_name(f.col("path"))).drop("path")

    # run the face detection function on each row
    detected_face_list_df = image_name_df.withColumn("detected_face_list", Face_Extraction_UDF("origin_image_name"))
    detected_face_df = detected_face_list_df.withColumn("extracted_face_image_name",
                                                        f.explode(f.col("detected_face_list"))).drop(
        "detected_face_list")
    return detected_face_df


def read_image_streaming(spark):
    image_input_folder_path = "/tmp/sparkcv/input"
    image_schema = spark.read.format("binaryFile").load(image_input_folder_path).schema
    raw_image_df_stream = spark.readStream.format("binaryFile").option("pathGlobFilter", "*.png") \
        .schema(image_schema) \
        .load(image_input_folder_path)
    return raw_image_df_stream


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


Face_Mask_Prediction_UDF = f.udf(lambda face_image_name: face_mask_prediction(face_image_name), BooleanType())


def get_face_coordinate_of_origin_image(face_image_name):
    x = face_image_name.split("_")[1][1:]
    y = face_image_name.split("_")[2][1:]
    w = face_image_name.split("_")[3][1:]
    h = face_image_name.split("_")[4][1:].split('.')[0]
    return int(x), int(y), int(w), int(h)


def integrate_face_mask_prediction(face_image_name, origin_image_name, has_mask):
    image_input_folder_path = "/tmp/sparkcv/input"
    final_output_path = "/tmp/sparkcv/output/final"
    # check if the image is already treated or not, if yes, it means it has multiple faces. and we just add new mask
    # prediction to untreated faces.
    # If not treated, we load image from
    treated_image_path = "{}/{}".format(final_output_path, origin_image_name)
    # If the image is treated, update the treated image
    if os.path.isfile(treated_image_path):
        image = cv2.imread(treated_image_path)
    else:
        # Get the untreated image from input
        image = cv2.imread("{}/{}".format(image_input_folder_path, origin_image_name))

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


Integrate_Face_Mask_Prediction_UDF = f.udf(
    lambda face_image_name, origin_image_name, has_mask: integrate_face_mask_prediction(face_image_name,
                                                                                        origin_image_name, has_mask))


def main():
    spark = SparkSession.builder \
        .master("local") \
        .appName("StreamingExample") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
        .getOrCreate()
    # Step1: read raw image as stream
    raw_image_df_stream = read_image_streaming(spark)
    # raw = raw_image_df_stream.writeStream \
    #     .format("console") \
    #     .option("truncate", "false") \
    #     .start()
    # Step2: detect faces from the raw image
    face_df_stream = detect_faces_streaming(raw_image_df_stream)
    # Step3: predict if they wear mask or not
    predict_mask_df_stream = face_df_stream.withColumn("has_mask",
                                                       Face_Mask_Prediction_UDF("extracted_face_image_name"))
    # step4: Integrate faces with tag to origin image
    complete_df = predict_mask_df_stream.withColumn("integration",
                                                    Integrate_Face_Mask_Prediction_UDF("extracted_face_image_name",
                                                                                       "origin_image_name",
                                                                                       "has_mask"))
    stream = complete_df.writeStream \
        .format("console") \
        .option("truncate", "false") \
        .start()
    stream.awaitTermination(200)
    stream.stop()


if __name__ == "__main__":
    main()
