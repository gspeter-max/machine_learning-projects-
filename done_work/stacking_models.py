
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName(' static').getOrCreate()
df_csv = spark.read.csv('/content/drive/MyDrive/ecommerce_churn_data.csv', header = True, inferSchema = True)
df_csv = df_csv.repartition(8)
df_csv.write.option('compression','snappy').parquet('/content/drive/MyDrive/ecommerce_churn_data.parquet')
df = spark.read.parquet('/content/drive/MyDrive/ecommerce_churn_data.parquet')

print(df.count() )

import numpy as np
from sklearn.preprocessing import TargetEncoder
df_pandas = df.toPandas()
df_pandas.isnull().sum()

df_pandas['Age'] = df_pandas['Age'].fillna(int(df_pandas['Age'].mean()))
df_pandas['AvgSessionLengthMinutes'] = df_pandas['AvgSessionLengthMinutes'].fillna(df_pandas['AvgSessionLengthMinutes'].mean())

def encoder(col, df):
    values = np.unique(df[col])
    index = np.arange(0,len(values) , 1)

    df[col] = df[col].replace(values, index)

encoder('Churn',df_pandas)

target_encoder = TargetEncoder()
df_pandas[['Gender', 'Location', 'ContractType']] = target_encoder.fit_transform(df_pandas[['Gender', 'Location', 'ContractType']],df_pandas['Churn'].values)
df_pandas['CustomerID'] = (df_pandas['CustomerID'] - min(df_pandas['CustomerID'])) + 1

mean_ = df_pandas['AvgSessionLengthMinutes'].mean()
std_ = df_pandas['AvgSessionLengthMinutes'].std()

higher_bound = mean_ + 3 * std_
lower_bound = mean_ - 3 * std_

out_free_df = df_pandas[(df_pandas['AvgSessionLengthMinutes'] >= lower_bound) & (df_pandas['AvgSessionLengthMinutes'] <= higher_bound)]
out_free_df['Age']  = out_free_df['Age'].astype(int)

from sklearn.model_selection import train_test_split
x = out_free_df.drop(['Churn','CustomerID','Age','Gender'],axis = 1)
y = out_free_df['Churn']
x_train,x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size= 0.2,stratify=y)


from sklearn.metrics import classification_report , confusion_matrix, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import  XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

predictors = [
    ('rf' , RandomForestClassifier(n_estimators = 200, random_state = 42 , max_depth = 5 )),
    ('xgb', XGBClassifier(eval_metric = 'auc')),
    ('lightgbm' , LGBMClassifier(random_state = 42))
]
base_ = LogisticRegression()

model = StackingClassifier(predictors,base_, cv = 5 ,n_jobs = -1)

model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:,1]

print(f' classification report : {classification_report(y_test, y_prediction)}')
print(f' confusion_matrix  : {confusion_matrix(y_test, y_prediction)} ')
print(f' roc_auc : {roc_auc_score(y_test, y_proba)}')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

mdoel_svc = SVC()
mdoel_svc.fit(x_train , y_train)

y_pred = mdoel_svc.predict(x_test)

print(f' classification report : {classification_report(y_test, y_pred)}')
print(f' confusion_matrix  : {confusion_matrix(y_test, y_pred)} ')
# print(f' roc_auc : {roc_auc_score(y_test, y_pred)}')
