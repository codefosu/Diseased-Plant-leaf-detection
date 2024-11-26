# Import Spark libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

# Set up Spark session and context
conf = SparkConf().setAppName("DistributedMNIST").setMaster("local[*]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
