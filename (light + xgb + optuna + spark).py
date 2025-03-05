from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('lightxgb').getOrCreate()
df = spark.read.csv('/content/sample_data/train.csv', header = True , inferSchema = True)
df.show()

from pyspark.ml.feature import OneHotEncoder , MinMaxScaler, StringIndexer, VectorAssembler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , roc_auc_score, confusion_matrix
from pyspark.ml import Pipeline
import numpy as np


education_indexer = StringIndexer( inputCols = ['education_level'], outputCols = ['education_level_'])
city_indexer = StringIndexer( inputCols = ['city'], outputCols = ['city_'])

one_hot_education = OneHotEncoder( inputCols = ['education_level_'] , outputCols = ['onehot_educ_encoder'])
one_hot_city = OneHotEncoder( inputCols = ['city_'], outputCols = ['onehot_city_encoder'])

vectorass = VectorAssembler( inputCols = ['onehot_educ_encoder','onehot_city_encoder','years_of_experience','salary','age'], outputCol= 'vector_scale')
min_max = MinMaxScaler( inputCol = 'vector_scale', outputCol = 'min_max_scaled')

pipeline = Pipeline(stages = [education_indexer ,city_indexer , one_hot_education, one_hot_city, vectorass , min_max])

df = pipeline.fit(df).transform(df)

df_pandas = df.select('loan_default','min_max_scaled').toPandas()
df_pandas['min_max_scaled'] = df_pandas['min_max_scaled'].apply(lambda x : x.toArray() if hasattr(x , 'toArray') else x)
# df_pandas['min_max_scaled'] = df_pandas['min_max_scaled'].apply(lambda x: x.toArray() if hasattr(x, "toArray") else x)
x = np.array(df_pandas['min_max_scaled'].tolist())
y = np.array(df_pandas['loan_default'])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.utils.class_weight import compute_class_weight 
import numpy as np

weight = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
class_weight_ = {key : value for key, value in zip(np.unique(y_train),weight)}
print(class_weight_)

xgb =XGBClassifier(learning_rate = 0.01, n_estimators = 200, max_depth = 5 , use_label_ecoder = False, eval_metric = 'auc',class_weight = class_weight_ )
xgb.fit(x_train , y_train)

y_predict = xgb.predict(x_test)
y_predict_proba = xgb.predict_proba(x_test)[:,1]

print(f"classification report : {classification_report(y_test, y_predict)}")
print(f"confusion matrix : {confusion_matrix(y_test, y_predict)}")
print(f"roc_auc_score : {roc_auc_score(y_test, y_predict_proba)}")

from lightgbm import LGBMClassifier

light_gbm = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 1010, max_depth = 3, learning_rate= 0.01, n_estimators = 200,class_weight = class_weight_)
x_train_ , x_test_ , y_train_, y_test_ = train_test_split(x,y,test_size = 0.2, random_state= 42)
light_gbm.fit(x_train_, y_train_)


y_predict_l = light_gbm.predict(x_test_)
y_predict_proba_l = light_gbm.predict_proba(x_test_)[:,1]

print(f"classification report : {classification_report(y_test_, y_predict_l)}")
print(f"confusion matrix : {confusion_matrix(y_test_, y_predict_l)}")
print(f"roc_auc_score : {roc_auc_score(y_test_, y_predict_proba_l)}")


import optuna 

'''
Learning rate
Number of estimators
Max depth
Subsample
Colsample_bytree

''' 
def objective(trial): 
    
    x_train_, x_val_, y_train_, y_val_ = train_test_split(x, y, test_size=0.2, random_state=42,stratify =y)
    from sklearn.utils.class_weight import compute_class_weight 
    import numpy as np

    weight_ = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    class_weight__ = {key : value for key, value in zip(np.unique(y_train),weight_)}

    params = {
        "learning_rate": trial.suggest_float('learning_rate', 0.0001, 0.4), 
        "n_estimators": trial.suggest_int('n_estimators', 200,500 ), 
        "max_depth": trial.suggest_int('max_depth', 3, 10), 
        "subsample": trial.suggest_float('subsample', 0.1, 1.0), 
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.1, 1.0), 
        "boosting_type": "gbdt", 
        'class_weight' : class_weight__
    }

    # Use proper train-test split here
    
    optuna_light = LGBMClassifier(**params)
    optuna_light.fit(x_train_, y_train_)

    y_predict_proba = optuna_light.predict_proba(x_val_)[:,1]
    score = roc_auc_score(y_val_, y_predict_proba)  # Evaluate on validation set, not train

    return score

study = optuna.create_study(direction= 'maximize')
study.optimize(objective , n_trials = 10)
best_par = study.best_params 
print(best_par)
upgrade_model = LGBMClassifier(**best_par)
upgrade_model.fit(x_train, y_train)  # Train only on training data



update_predict = upgrade_model.predict(x_test)
update_predict_proba = upgrade_model.predict_proba(x_test)[:,1]

print(f"classification report : {classification_report(y_test, update_predict)}")
print(f"confusion matrix : {confusion_matrix(y_test, update_predict)}")
print(f"roc_auc_score : {roc_auc_score(y_test, update_predict_proba)}")


''' test codes ''' 

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('lightxgb').getOrCreate()
df_test = spark.read.csv('/content/sample_data/test.csv', header = True , inferSchema = True)
df_test.show()
from pyspark.ml.feature import OneHotEncoder , MinMaxScaler, StringIndexer, VectorAssembler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , roc_auc_score, confusion_matrix
from pyspark.ml import Pipeline
import numpy as np


education_indexer = StringIndexer( inputCols = ['education_level'], outputCols = ['education_level_'])
city_indexer = StringIndexer( inputCols = ['city'], outputCols = ['city_'])

one_hot_education = OneHotEncoder( inputCols = ['education_level_'] , outputCols = ['onehot_educ_encoder'])
one_hot_city = OneHotEncoder( inputCols = ['city_'], outputCols = ['onehot_city_encoder'])

vectorass = VectorAssembler( inputCols = ['onehot_educ_encoder','onehot_city_encoder','years_of_experience','salary','age'], outputCol= 'vector_scale')
min_max = MinMaxScaler( inputCol = 'vector_scale', outputCol = 'min_max_scaled')

pipeline = Pipeline(stages = [education_indexer ,city_indexer , one_hot_education, one_hot_city, vectorass , min_max])

df_test = pipeline.fit(df_test).transform(df_test)

df_pandas_test = df_test.select('loan_default','min_max_scaled').toPandas()
df_pandas_test['min_max_scaled'] = df_pandas_test['min_max_scaled'].apply(lambda x : x.toArray() if hasattr(x , 'toArray') else x)
# df_pandas['min_max_scaled'] = df_pandas['min_max_scaled'].apply(lambda x: x.toArray() if hasattr(x, "toArray") else x)
x_testing = np.array(df_pandas_test['min_max_scaled'].tolist())
y_testing = np.array(df_pandas_test['loan_default'])

final_predict = upgrade_model.predict(x_testing)
final_predict_proba = upgrade_model.predict_proba(x_testing)[:,1]

print(f"classification report final  : {classification_report(y_testing, final_predict)}")
print(f"confusion matrix final : {confusion_matrix(y_testing, final_predict)}")
print(f"roc_auc_score final : {roc_auc_score(y_testing, final_predict_proba)}")

''' 
result : 
classification report final  :               precision    recall  f1-score   support

           0       0.70      0.86      0.77       139
           1       0.34      0.16      0.22        61

    accuracy                           0.65       200
   macro avg       0.52      0.51      0.50       200
weighted avg       0.59      0.65      0.61       200

confusion matrix final : [[120  19]
 [ 51  10]]
roc_auc_score final : 0.5302512088689704
''' 
