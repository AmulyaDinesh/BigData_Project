import joblib       #lightweight pipelining in Python
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
import numpy as np

sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")
ssc = StreamingContext(sc, 1)
spark=SparkSession(sc)
data = ssc.socketTextStream("localhost", 6100) 

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

mlp = MLPClassifier(hidden_layer_sizes = (150,100,150),max_iter = 300,activation = 'relu',solver = 'adam',random_state = 1)
bernoulli = BernoulliNB()
passive_classifier = PassiveAggressiveClassifier(C = 0.5,random_state = 5)
def fit_mlp(x_train,y_train):
  mlp.partial_fit(x_train,y_train,classes = np.unique(y_train))
def fit_bernoulli(x_train,y_train):
  bernoulli.partial_fit(x_train,y_train,classes = np.unique(y_train))
def fit_passive(x_train,y_train):
  passive_classifier.partial_fit(x_train,y_train,classes = np.unique(y_train))
def readMyStream(rdd): 
		if(len(rdd.collect())>0):
		  df = spark.read.json(rdd)
		  df = df.drop('Subject')
		  df = df_preprocessing(df)
		  x = np.array(df.select('Hash').collect())
		  x = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
		  y = np.array(df.select('Status').collect())
		  y = y.flatten()
		  fit_mlp(x,y)
		  fit_bernoulli(x,y)
		  fit_passive(x,y) 
		  joblib.dump(mlp,'mlp.pkl')
		  joblib.dump(bernoulli,'bernoulli.pkl')
		  joblib.dump(passive_classifier,'passive.pkl')

data.foreachRDD(lambda rdd: readMyStream(rdd))
ssc.start()
ssc.awaitTermination()