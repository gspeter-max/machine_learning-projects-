from pyspark.sql import SparkSession
from pyspark.ml.features import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


spark = SparkSession.builder.appName('data_load').getOrCreate()
df = spark.read.option('header', True).option('inferSchema', True).csv('/content/sample_data/mnist_test.csv')

num_partition = max(100, df.rdd.getNumPartitions()) 
df = df.repartition(num_partition)

df.write.mode('overwrite').option('mergeSchema', True).option('compression', 'snappy').parquet('/content/sample_data/mnist_test.parquet')
df_parquet = spark.read.parquet('/content/sample_data/mnist_test.parquet')

df_parquet = df_parquet.fillna({'col_name': 0})  
numeric_features = [c for c, dtype in df_parquet.dtypes if dtype in ('int', 'double')]
vector = VectorAssembler(inputCols=numeric_features, outputCol='features_1')
df_vectors = vector.transform(df_parquet)

lr = LogisticRegression(featuresCol='features_1', labelCol='label_features')  
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.2, 0.1, 0.4])\
    .addGrid(lr.maxIter, [100, 200, 300])\
    .build()

crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=3 
)
best_model = crossval.fit(df_vectors)
df_parquet.show(10)
