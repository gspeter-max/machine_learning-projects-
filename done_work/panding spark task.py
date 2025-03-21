columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('GPT-Challange').getOrCreate()
df = spark.read.csv('/content/drive/MyDrive/adult.csv',header= False,inferSchema = True)
df = df.toDF(*columns)
df = df.repartition(100)

df.write.option('compression','snappy').mode('overwrite').parquet('/content/drive/MyDrive/adult.parquet')


parquet_df = spark.read.parquet('/content/drive/MyDrive/adult.parquet')
parquet_df.createOrReplaceTempView('aduil')
df = spark.sql(
    '''
    select * ,
        case when cast( trim(income) as string)  = '>50K' then 1 else 0 end as income_binary
    from aduil;
    '''
)
parquet_df  = df.drop('income')

'''

shap + randomforst + train_again
'''




from pyspark.ml.feature import VectorAssembler , StringIndexer , StandardScaler , OneHotEncoder
from pyspark.ml import Pipeline


columns = parquet_df.columns
lists = ['capital_gain','capital_loss','hours_per_week','age','income_binary']


for col in lists:
    columns.remove(col)


output_list = [out_col + '_onehot' for out_col in columns]
output_index = [out_col + 'indexer' for out_col in columns]



index = StringIndexer(inputCols= columns, outputCols = output_index)
one_hots = OneHotEncoder(inputCols = output_index, outputCols = output_list )


pipeline = Pipeline(stages= [index, one_hots])
df = pipeline.fit(parquet_df).transform(parquet_df)

pandas_remove = ['workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex',  'native_country']

train_df, test_df = df.randomSplit([0.8,0.2], seed = 42)
train_pandas_df  = train_df.toPandas() 
train_pandas_df = train_pandas_df.drop(columns = ['workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex',  'native_country'])

from sklearn.ensemble import  RandomForestClassifier 
import shap 

shaping_model  = RandomForestClassifier(
    n_estimators = 100, 
    random_state = 42
)

explainer = shap.TreeExplainer(shaping_model, train_pandas_df) 
shap_values = explainer.shap_values(train_pandas_df)


