from pyspark.sql import SparkSession 

spark = SparkSession.builder.appName("Custom Transformation Function").getOrCreate() 
df = spark.read.csv("data.csv", header=True, inferSchema=True) 
df = df.repartition(8) 

df.write.options('compression', 'snappy').parquet("data.parquet") 

df_parquet = spark.read.parquet("data.parquet") 
''' 
continue and read from 
https://www.toptal.com/spark/apache-spark-optimization-techniques

make sure learn more 
''' 

