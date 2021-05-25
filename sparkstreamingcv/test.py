import itertools
from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def word_count_streaming(spark):
    # use nc -lk 9999 to create a socket
    # Create DataFrame representing the stream of input lines from connection to localhost:9999
    lines = spark \
        .readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    # Split the lines into words
    words = lines.select(
        f.explode(
            f.split(lines.value, " ")
        ).alias("word")
    )
    # Generate running word count
    wordCounts = words.groupBy("word").count()

    # Start running the query that prints the running counts to the console
    # delta only support append and complete outputMode
    stream = wordCounts \
        .writeStream \
        .outputMode("complete") \
        .format("delta") \
        .option("checkpointLocation", "/tmp/sparkcv/delta/events/_checkpoints/word_count") \
        .start("/tmp/sparkcv/delta/events")  # output path of the data in delta "format"

    stream.awaitTermination(100)
    stream.stop()


def read_delta_streaming(spark):
    stream = spark.readStream.format("delta").load("/tmp/sparkcv/delta/events").writeStream \
        .format("console") \
        .start()
    stream.awaitTermination(100)
    stream.stop()


def read_image_streaming(spark):
    image_input_folder_path="/tmp/sparkcv/input"
    image_schema = spark.read.format("binaryFile").load(image_input_folder_path).schema
    stream = spark.readStream.format("binaryFile").option("pathGlobFilter", "*.png") \
        .schema(image_schema) \
        .load(image_input_folder_path).writeStream.format("console").start()
    stream.awaitTermination(100)
    stream.stop()


def main():
    spark = SparkSession.builder \
        .master("local") \
        .appName("StreamingExample") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
        .getOrCreate()

    # read_delta_streaming(spark)
    read_image_streaming(spark)

if __name__ == "__main__":
    main()
