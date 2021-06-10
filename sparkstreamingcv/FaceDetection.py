from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_fun
import cv2
from pyspark.sql.functions import udf

spark = SparkSession.builder \
    .master("local") \
    .appName("FaceDetection") \
    .getOrCreate()


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

    # extract faces from input image
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = img[y:y + h, x:x + w]
        extracted_face_output_path = faces_output_path + image_name[:-4] + "_x" + str(x) + "_y" + str(y) + "_w" + str(
            w) + "_h" + str(h) + ".png"
        cv2.imwrite(extracted_face_output_path, crop)

    return str(len(faces))


# This function can extract a file name from a full file path
# path is the column name where stores the full file path
def extract_file_name(path):
    return sql_fun.substring_index(path, "/", -1)


# create a udf which calls face_extraction function
Face_Extraction_UDF = udf(lambda image_name: face_extraction(image_name))


def detect_faces_in_batch(image_input_folder_path):
    # read images as df from a folder
    image_schema = spark.read.format("binaryFile").load(image_input_folder_path).schema
    raw_images_df = spark.read \
        .format("binaryFile") \
        .schema(image_schema) \
        .load(image_input_folder_path)
    raw_images_df.show()

    # get the image name df
    image_name_df = raw_images_df \
        .select("path") \
        .withColumn("image_name", extract_file_name(sql_fun.col("path"))).drop("path")
    image_name_df.show()

    # run the face detection function on each row
    detected_face_df = image_name_df.withColumn("detected_face_number", Face_Extraction_UDF("image_name"))
    detected_face_df.show()


def main():
    image_input_folder_path = "/tmp/sparkcv/input/"

    # face_extraction(image_input_folder_path, "maksssksksss331.png", cascade_model_path, faces_output_path)
    detect_faces_in_batch(image_input_folder_path)


if __name__ == "__main__":
    main()


#
# with fs.open(image_path,mode='rb') as f:
#    raw = f.read()
#    # print(raw)
#    nparr = np.fromstring(raw, np.uint8)
#    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#    print(nparr)
#    cv2.imshow('Final_Image',img)