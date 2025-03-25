from pyspark.sql import  SparkSession 

spark = SparkSession.builder.appName('shap_features_selection').getOrCreate() 
df= spark.read.csv('data.csv', header=True, inferSchema=True)

# if you have categortical features 

from pyspark.ml.feature import StringIndexer, OneHotEncoder 

indexer = StringIndexer(inputCol = 'catgorical_feature', outputCol = 'catgorical_feature_index') 
one_hot = OneHotEncoder(inputCol= 'categorical_feature_index', outputCol= 'categorical_feature') 

from pyspark.ml.pipeline import Pipeline 

pipelines= Pipeline(stages= [indexer , one_hot]) 
df = pipelines.fit(df).transform(df) 

# standardization 

from pyspark.ml.feature import VectorAssembler , StandardScaler 

vector  = VectorAssembler(inputCols= ['list_of_column'] , outputCol= 'combine_list') 
df_transform = vector.transform(df) 

std = StandardScaler(inputCol= 'combine_features', outputCol= 'std_features', withMean= True, withStd= True)
df_std = std.fit(df_transform).transform(df_transform) 

''' shap not allow spark models ''' 
df_pandas = df_std.select('std_features','target').toPandas() 

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 

x = df_pandas['std_features'] 
y = df_pandas['target'] 

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42) 

rf = RandomForestClassifier(
    n_estimators= 200,
    random_state = 42
)
rf.fit(x_train, y_train) 


import shap

# if you workign with linear model that use ( shap.LinearExplainer ) 
explainer = shap.TreeExplainer(rf, x_train) 
shap_values = explainer.shap_values(x_train) 

shap_values = np.mean(shap_values, axis = 0)[:,1] # if you do binary task 
top_k = int(input('how many important features you need')) 
top_k_features = np.argsort(shap_values)[::-1][:top_k]

print(f' important feattures  : {df.columns[top_k_features]}') 
x_train_top = x_train[:,top_k_features] 
x_test_top = x_test[:top_k_features] 

final_model = RandomForestClassifier(
    n_estimators= 200, 
    random_state = 42
)
final_model.fit(x_train_top,y_train) 

y_pred = final_model.predict(x_test_top) 

from sklearn.metrics import classification_report 

print(f' classification report : {classification_report(y_test, y_pred)}') 
