import json
from pyspark.sql import Row
from pyspark.sql.types import StructType,StructField, StringType
import pandas as pd

import time
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import udf,variance
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
rdd1=0
sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")
#streaming the context
ssc = StreamingContext(sc, 1) 
spark=SparkSession(sc)

# list of all rdd to read the stream
dataset = ssc.socketTextStream("localhost", 6100)
def readMyStream(rdd): 
		
		#r = rdd.map(lambda x: json.loads(x))
		if(len(rdd.collect()) > 0):
		  df = spark.read.json(rdd) #converting to json
		  df.show()
		    

dataset.foreachRDD(lambda rdd: readMyStream(rdd))
ssc.start()
ssc.awaitTermination()
