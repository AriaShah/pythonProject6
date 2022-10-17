import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import findspark
findspark.init()
import sys
import os
#os.environ['SPARK_HOME'] = "C:/Spark/spark-3.3.0-bin-hadoop3"
#sys.path.append("C:/Spark/spark-3.3.0-bin-hadoop3/python")
#sys.path.append('C:/Spark/spark-3.3.0-bin-hadoop3/python/pyspark')

df = pd.read_csv('C:/Users/HP/Desktop/ir4.csv')

# Create the Session
spark = SparkSession.builder \
    .master("local") \
    .appName("kmeans") \
    .getOrCreate()

sc = spark.sparkContext

df1 = spark.createDataFrame(df)
vecAssembler = VectorAssembler(inputCols=["lat", "lng"], outputCol="features")
new_df = vecAssembler.transform(df1)
new_df.show()

kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(new_df.select('features'))
transformed = model.transform(new_df)
transformed.show()