import itertools

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .master("local") \
    .appName("FaceDetection") \
    .getOrCreate()

source_data = [
    ("toto", "toto_1", "00", 1),
    ("titi", "titi_1", "00", 2),
    ("tata", "titi_1", "00", 3),
    ("toto", "toto_2", "00", 4),
    ("toto", "toto_3", "00", 5),
]

source_df = spark.createDataFrame(source_data, ["name", "key", "content", "value"])
source_df.show()

# solution one: use key value map structure
target_df = source_df.withColumn("target", F.create_map("key", "value"))
target_df.show()
target_df.printSchema()
dfMap = target_df.groupby("name").agg(F.collect_list("target").alias("list"))
dfMap.show(5, False)


def flat_list(source_list):
    return list(itertools.chain(*source_list))


Flat_List_UDF = F.udf(lambda list: flat_list(list))

dfFlatMap = dfMap.withColumn("flat_list", Flat_List_UDF("list"))
dfFlatMap.show()

# solution two use user define struct type, solution two is better.
df_new = source_df.groupBy("name").agg(F.collect_list(
    F.struct(
        *[F.col("key").alias("t.key"), F.col("content").alias("t.content"), F.col("value").alias("t.value")])).alias(
    "target"))

df_new.show(5, False)
df_new.printSchema()


def show_content(varList):
    res = []
    for var in varList:
        name = var[0]
        content = var[1]
        value = var[2]
        res.append(name + content + str(value))
    return res


Show_Content_UDF = F.udf(lambda varList: show_content(varList))

test=df_new.withColumn("test", Show_Content_UDF("target"))

test.show(5,False)
# df_new = source_df.withColumn(
#     'target',
#     F.struct(*[F.col('key').alias('t_key'),F.col("content").alias('t_content'), F.col("value").alias('t_value')])
# )
#
# df_new.show()
# df_new.printSchema()
# df_new_collected = target_df.groupby("name").agg(F.collect_list("target").alias("list"))
# df_new_collected.show(5, False)
#
# df_new_collected.printSchema()
# def get_value(keyValueList):
#     for keyValue in keyValueList:
#         keyValue.get


"""
scala code work with column struct type

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions._

val df = Seq(
  ("str", 1, 0.2)
).toDF("a", "b", "c").
  withColumn("struct", struct($"a", $"b", $"c"))

// UDF for struct
val func = udf((x: Any) => {
  x match {
    case Row(a: String, b: Int, c: Double) => 
      s"$a-$b-$c"
    case other => 
      sys.error(s"something else: $other")
  }
}, StringType)

df.withColumn("d", func($"struct")).show

"""
