

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from kneed import DataGenerator, KneeLocator
import pyspark
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession
#df = pd.read_csv('C:/Users/HP/Desktop/ir.csv')
#df.dropna(axis=0,how='any',subset=['population'],inplace=True)


# Variable with the Longitude and Latitude
#X_weighted =df.loc[:,['city','lat','lng','population']]

#X_weighted.head(4)

#lat_long = X_weighted[X_weighted.columns[1:4]]
#pop_size = X_weighted[X_weighted.columns[3]]
#sample_weight = pop_size

spark = SparkSession.builder\
    .master("local")\
    .appName("Kmeans")\
    .getOrCreate()

sc = spark.sparkContext


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

if __name__ == "__main__":
    #line = spark.read.csv('C:/Users/HP/Desktop/ir3.csv')
    lines = sc.textFile('C:/Users/HP/Desktop/ir4.csv')
    data = lines.map(parseVector)
    data.top(5)
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans.train(data, k)
        kmeans.setWeightCol("population")
        kmeans.setMaxIter(10)
        kmeans.fit(data)
        sse.append(kmeans.getDistanceMeasure())


# Elbow plot
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

