from pyspark.sql import DataFrame
from pyspark.sql import  SparkSession


spark = SparkSession.builder.appName("shaping").getOrCreate()
df = spark.read.csv('/content/drive/MyDrive/classification_data.csv',header= True , inferSchema= True)

def compute_features_for_pyspark(df: DataFrame,target_name : str, categorical_features : str, model_type : str ,n:int):

    from sklearn.ensemble import RandomForestClassifier
    import shap
    from sklearn.model_selection import train_test_split
    import numpy as np 

    df_pandas  = df.toPandas()
    x = df_pandas.drop(columns = target_name)

    if categorical_features and categorical_features is not  None:
        x = x.drop(categorical_features , axis = 1 )

    y = df_pandas[target_name]

    model = RandomForestClassifier(
        n_estimators = 200,
        random_state = 42
    )
    model.fit(x,y)
    if model_type.lower() == 'tree':
        explainer = shap.TreeExplainer(model,x)
    elif model_type.lower() == 'linear':
        explainer = shap.LinearExplainer(model,x)
    else:
        explainer = shap.Explainer(model, x)

    shap_values = explainer.shap_values(x)

    if model_type.lower() == 'tree':
        shap_values = np.mean(shap_values, axis = 0 )[:,1]
    else :
        shap_values= np.mean(shap_values, axis = 0)

    features_index = np.argsort(shap_values)[:-1][:n]
    important_features = x.columns[features_index]
    return list(important_features)

shap_features = compute_features_for_pyspark(df = df  ,target_name = 'target',categorical_features= None,model_type= 'tree' ,n = 7)

from pyspark.ml.classification  import RandomForestClassifier
from pyspark.ml.feature import  VectorAssembler

vector = VectorAssembler(inputCols =  shap_features, outputCol = 'features')
df_transform = vector.transform(df)

train_df , test_df = df_transform.randomSplit([0.80,0.20], seed = 42)

model = RandomForestClassifier(
    featuresCol = 'features',
    labelCol = 'target',
    numTrees = 10
)

model = model.fit(train_df)
prediction = model.transform(test_df) 

prediction_features = prediction.select('target','prediction').toPandas() 
y_test = prediction_features['target']
predict = prediction_features['prediction']

from sklearn.metrics import classification_report 

print(f'classification report : {classification_report(y_test, predict)}')


