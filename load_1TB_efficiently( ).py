''' mode ( overwrite) that is simply say that if that exist deleted that and create new with new content

like : --> df.write.mode('overwrite').option('header' ,True).csv('folder_name')

if you lode csv file that time you use (header is Ture ) you say that time include first row as a number of colums the code of load csv is

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ai_data_preocessing').getOrCreate()

df = spark.read.csv('path' , [[[[header = True]]]], inferSchema = True ) ( inferSchema meaning typo of column is detected automatically )
you see that highlighed part


'''
from pyspark.sql import  SparkSession

spark = SparkSession.builder.appName('ai_data_processing').getOrCreate()

df = spark.read.csv('/content/linearly_separable_classification.csv',header = True, inferSchema = True)
df.repartition(100)

# df.write.parquet('/content/linearly_separable_classification.parquet')



'''
if your partitioning have difference parition table have different column then you merage out that using

df_merge = spark.read.option('mergeSchema', 'True').parquet('/content/sample_data/mnist_test.parquet')

if you need both (merge + partitioning agian ) you use

df.coalesce(1000)
that is not shuffle the data and if you use that df.repartition() that is shuffle the data that reason that slower than coalesce

i know in (sql ) that (coalesce) is used for ignore  null values
'''

df_parquet = spark.read.parquet('/content/linearly_separable_classification.parquet')
df_parquet.coalesce(1000)

''' 
if yours table have any null values remove that 

df_parquet.fillna({
    'colname ' ; sum(col('colname')) #  you also compute the avg of that 
})

using 
form pyspark.sql.functions import avg 


you make any columns 

df_parquet = df_parquet.withColumns('new_Column_name',col('amount')*3)
''' 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler  

vector = VectorAssembler(inputCols = df_parquet.columns , outputCol = 'vector_featurs')
df_temp = vector.transform(df)

lr = LogisticRegression(featuresCol = 'vector_featurs', labelCol = 'label')
model = lr.fit(df_temp)

df_parquet = df_parquet.coalesce(1000) 
spark.config.set('spark.sql.shuffle.partitions',500) # becuase initial that is 250 


