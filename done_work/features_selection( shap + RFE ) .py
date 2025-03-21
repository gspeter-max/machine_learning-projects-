from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('baysian_things').getOrCreate()
df = spark.read.csv('/content/drive/MyDrive/raw_fraud_detection_dataset.csv',header = True, inferSchema = True)


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('genius').getOrCreate()
df = spark.read.csv('/content/drive/MyDrive/raw_fraud_detection_dataset.csv',header = True, inferSchema = True)

df_pandas = df.toPandas()
df_pandas = df_pandas.dropna()
x = df_pandas.drop(columns= ['Fraud_Label','Timestamp'])
y = df_pandas['Fraud_Label']

from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import  train_test_split
import shap
from sklearn.preprocessing import  LabelEncoder
encoder = LabelEncoder()
for col in ['Location','Device_Type','Merchant_Type']:
    x[col] = encoder.fit_transform(x[col])
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size= 0.3,random_state= 42)
model  = RandomForestClassifier(
    n_estimators = 200,
    random_state = 42, 
    class_weight = 'balanced'
)

model.fit(x_train , y_train)
prediction = model.predict(x_test)
from sklearn.metrics import classification_report 

print(f'classification report : {classification_report(y_test, prediction)}')


def features_selections(model,x_train,target,n, axis = 0): 
    
    create_study  = shap.Explainer(model, x_train )
    shap_values = create_study.shap_values(x_train)

    
    import numpy as np
    
    if len(np.unique(target)) == 2: 
        import_shap_values = np.mean(shap_values, axis = axis)[:,1]
    else: 
        import_shap_values = np.mean(shap_values, axis = axis)
    
    
    sort_index = np.argsort(import_shap_values)[:-1]
    shap_features  = x_train.columns[sort_index][:n]
    from sklearn.feature_selection import RFE

    features_elimination = RFE(estimator=model, n_features_to_select=n)
    mask = features_elimination.fit_transform(x_train,target) 

    rfe_features = np.array(x_train.columns)[features_elimination.support_]
    final_features = [] 

    for i in range(n): 
        if rfe_features[i] in shap_features: 
            final_features.append(rfe_features[i])
    if not final_features: 
        final_features = rfe_features
    print(final_features)
    return final_features

final_features = features_selections(model,x_train,y_train,5,0) 
model_rf = RandomForestClassifier(
    n_estimators = 200,
    random_state = 42, 
    class_weight = 'balanced'
)
model_rf.fit(x_train[final_features],y_train)
fe_predict = model_rf.predict(x_test[final_features])

print(f'classification report : { classification_report(y_test,fe_predict)}')

df_re = spark.read.csv('/content/drive/MyDrive/regression_dataset.csv',header = True, inferSchema = True)

df_pandas_re = df_re.toPandas()
df_pandas_re = df_pandas_re.dropna()
from sklearn.model_selection import train_test_split

x_re = df_pandas_re.drop('Target', axis = 1)
y_re = df_pandas_re['Target']

from sklearn.preprocessing import  LabelEncoder
encoder = LabelEncoder()

x_re['Feature4'] = encoder.fit_transform(x_re['Feature4'])

x_train_re , x_test_re , y_train_re , shap_valuesy_test_re  = train_test_split(x_re, y_re, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestRegressor

before_fe   = RandomForestRegressor(
    n_estimators= 200, 
    random_state =42 ,)

before_fe.fit(x_train_re, y_train_re)
before_predict  = before_fe.predict(x_test_re)

from sklearn.metrics import mean_squared_error
print(f'accuracy_score task before : {mean_squared_error(y_test_re,before_predict)}')

re_final_features = features_selections(before_fe, x_train_re, y_train_re,2,0)
after_fe = RandomForestRegressor(
    n_estimators= 200, 
    random_state =42 , 
)
print(re_final_features)
print(x_train_re[re_final_features])
after_fe.fit(x_train_re[re_final_features], y_train_re)
after_predict = after_fe.predict(x_test_re[re_final_features])

print(f'accuracy_score taks after  : {mean_squared_error(y_test_re, after_predict)}')
