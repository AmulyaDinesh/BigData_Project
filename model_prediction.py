import time
import re       #regular expression; specifies a set of strings that matches it
import joblib
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import udf,variance
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import json
from pyspark.sql import Row
from pyspark.sql.types import StructType,StructField, StringType
from pyspark.ml.feature import HashingTF, IDF
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import isnan, when, count, col,lit
from pyspark.sql import functions as F
from pyspark.ml.feature import StopWordsRemover,StringIndexer,Tokenizer,VectorAssembler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron,SGDClassifier,PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.cluster import Birch
import numpy as np
import matplotlib.pyplot as plt

sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")
ssc = StreamingContext(sc, 1)
spark=SparkSession(sc)
data = ssc.socketTextStream("localhost", 6100) 
mlp = joblib.load('mlp.pkl')
bernoulli = joblib.load('bernoulli.pkl')
passive = joblib.load('passive.pkl')

def df_preprocessing(df):
   df = df.na.drop()
   for col in df.columns:
    df = df.withColumn(col, F.lower(F.col(col))) 
   df = Tokenizer(inputCol = "Message",outputCol = "Tokens").transform(df)
   df = StopWordsRemover(inputCol = "Tokens",outputCol = "Words").transform(df)
   df = HashingTF(numFeatures=df.count(), inputCol="Tokens", outputCol='Hash').transform(df)
   df = StringIndexer(inputCol = 'Spam',outputCol = 'Status' , handleInvalid = 'keep').\
        fit(df).transform(df)
   
   return df
def mlp_predict(x,y):
  mlp_predictions = mlp.predict(x)
  mlp_score = accuracy_score(mlp_predictions,y)
  acc_mlp = open("accuracyMLP.txt","a")
  acc_mlp.write("{}\n".format(mlp_score))
  acc_mlp .close()
  return mlp_predictions
def bernoulli_predict(x,y):
  bernoulli_predictions = bernoulli.predict(x)
  bernoulli_score = accuracy_score(bernoulli_predictions,y)
  acc_bernoulli = open("accuyracyBernoulli.txt","a")
  acc_bernoulli.write("{}\n".format(bernoulli_score))
  acc_bernoulli.close()
  return bernoulli_predictions
  
def passive_predict(x,y):
  passive_predictions = passive.predict(x)
  acc_passive = open("accuracyPassive.txt","a")
  passive_score = accuracy_score(passive_predictions,y) 
  acc_passive.write("{}\n".format(passive_score))
  acc_passive.close()
  return passive_predictions
  
def readMyStream(rdd): 
		if(len(rdd.collect())>0):
		  df = spark.read.json(rdd)
		  df = df.drop('Subject')
		  df = df_preprocessing(df)
		  x = np.array(df.select('Hash').collect())
		  x = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
		  y = np.array(df.select('Status').collect())
		  y = y.flatten()
		  mlp_prediction = mlp_predict(x,y)
		  bernoulli_prediction = bernoulli_predict(x,y)
		  passive_prediction = passive_predict(x,y)
		  print("Confusion Matrix")
		  print(f"MLP :\n {classification_report(mlp_prediction,y)}")
		  print(f"Bernoulli NB :\n {classification_report(bernoulli_prediction,y)}")
		  print(f"Passive Agressive :\n {classification_report(passive_prediction,y)}")
		  print("Predictions")
		  print(f"MLP prediction {mlp_prediction}\n")
		  print(f"Bernoulli prediction {bernoulli_prediction}\n"),
		  print(f"Passive predcition {passive_prediction}"),
		  print("Batch predicted")
		 

data.foreachRDD(lambda rdd: readMyStream(rdd))
ssc.start()
ssc.awaitTermination()