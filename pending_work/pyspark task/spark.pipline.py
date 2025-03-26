from pyspark.sql import  SparkSession 

spark = SparkSession.builder.appName('pipeline').getOrCreate() 

df_csv = spark.read.csv('/workspaces/machine_learning-projects-/pending_work/pyspark task/small_media_data.csv',
                        header = True, inferSchema = True)

num_partitions = df_csv.rdd.getNumPartitions()  

df_csv = df_csv.repartition(num_partitions + 10)
df_csv.write.option('compression','snappy').mode('overwrite').parquet('/workspaces/machine_learning-projects-/pending_work/pyspark task/small_media_data.parquet') 

df_parquet = spark.read.parquet('/workspaces/machine_learning-projects-/pending_work/pyspark task/small_media_data.parquet')


'''
# if you do that in sql 

df_parquet.createOrReplaceTempView('media_table')
df_parquet = spark.sql(
    """
        with temp_temp as (
            select 
                post_id, 
                user_id, 
                cast(timestamp as date) as dates, 
                cast(date_format(timestamp,"M") as int)  as months , 
                likes, 
                case when likes > 500 then 1 else 0 end as engagement_score, 
                case when media_type = 'image' then 1 
                    when media_type = 'text' then 2 
                    else 3 end as media_type_encoded

            from media_table 
        )
        select * from temp_temp;
    """
)
'''

from pyspark.sql.functions import when,cast , col , date_format,cast 
from pyspark.sql.types  import DateType  

df_parquet = df_parquet.withColumn('timestamp' , col('timestamp').cast(DateType())).withColumnRenamed('timestamp','date')
df_parquet = df_parquet.withColumn('months' , date_format('date', 'M').cast('int'))
df_parquet= df_parquet.withColumn('engagement_score', when(col('likes') > 500 ,1).otherwise(0))
df_parquet = df_parquet.withColumn('media_type',when(col('media_type') == 'image', 1).when(col('media_type') == 'video',2).otherwise(3)).withColumnRenamed('media_type', 'media_type_encoded')
df_parquet.show()


train , test = df_parquet.randomSplit([0.8,0.2], seed= 42)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import  RandomForestClassifier

train = train.drop('post_id','user_id','date')

column_list = train.columns
column_list.remove('engagement_score')

vector = VectorAssembler(
    inputCols = column_list,
    outputCol = 'vector_list'
)
train_transformed = vector.transform(train)
test_transformed = vector.transform(test)

model = RandomForestClassifier(featuresCol='vector_list', labelCol='engagement_score') 
model = model.fit(train_transformed) 

from pyspark.ml.evaluation import  BinaryClassificationEvaluator

prediction = model.transform(test_transformed)
prediction.show()
evaluator = BinaryClassificationEvaluator(
    labelCol = 'engagement_score',
    rawPredictionCol = 'rawPrediction', 
    metricName = 'areaUnderROC'
)
roc = evaluator.evaluate(prediction)
print(f'roc : {roc}')

