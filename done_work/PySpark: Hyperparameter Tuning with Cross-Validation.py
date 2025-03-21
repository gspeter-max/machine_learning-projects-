from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Name').getOrCreate()

df = spark.read.csv('/content/drive/MyDrive/classification_data.csv', header = True, inferSchema = True)
df.columns

from pyspark.ml.pipeline import  Pipeline
from pyspark.ml.classification import  LogisticRegression
from pyspark.ml.tuning import  ParamGridBuilder,CrossValidator
from pyspark.ml.evaluation import  BinaryClassificationEvaluator
from pyspark.ml.feature import  VectorAssembler


vector = VectorAssembler(inputCols= ['_c0',
 'Feature_0',
 'Feature_1',
 'Feature_2',
 'Feature_3',
 'Feature_4',
 'Feature_5',
 'Feature_6',
 'Feature_7',
 'Feature_8',
 'Feature_9'], outputCol = 'features_vector')

evaluator  = BinaryClassificationEvaluator(labelCol= 'target',)
lr = LogisticRegression(featuresCol = 'features_vector')
params = ParamGridBuilder().baseOn(
    {
        lr.labelCol : 'target'
    }
).baseOn([
    lr.predictionCol,'predict'
]).addGrid(
    lr.regParam,[0.1,0.3]
).addGrid(
    lr.maxIter, [12,10]
).addGrid(
    lr.elasticNetParam,[0,1]
).build()

validator  = CrossValidator(
    estimator= lr ,
    estimatorParamMaps = params,
   evaluator = evaluator
)
df_transform = vector.transform(df)
preformance = validator.fit(df_transform)

train_df ,test_df = df_transform.randomSplit([0.8,0.2], seed = 42)
best_model  = preformance.bestModel
best_regparams = best_model._java_obj.getRegParam()
best_elastic = best_model._java_obj.getElasticNetParam()


made_model = LogisticRegression(featuresCol = 'features_vector' , labelCol = 'target', elasticNetParam= best_elastic,
                                        regParam = best_regparams)
made_model = made_model.fit(train_df)
prediction = made_model.transform(test_df)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'probability',labelCol= 'target')

auc = evaluator.evaluate(prediction)
print(auc)



