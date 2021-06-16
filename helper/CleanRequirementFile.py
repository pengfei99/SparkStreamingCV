from pyspark.sql import SparkSession


def clean_poetry_generated_requirement_file(spark, input_file_path, output_file_path):
    raw = spark.read.option("delimiter", ";").csv(input_file_path).toDF("dependency", "python-version")
    # raw.show(5, False)
    dep = raw.select("dependency")
    dep.show(50, False)
    dep.coalesce(1).write.format("com.databricks.spark.csv").option("header", "false").save(output_file_path)


def main():
    spark = SparkSession.builder.master("local").appName("test").getOrCreate()
    input_file_path = "/freshpy38-requirements.txt"
    output_file_path = "/tmp/requirement.txt"
    clean_poetry_generated_requirement_file(spark,input_file_path,output_file_path)

if __name__ == "__main__":
    main()
