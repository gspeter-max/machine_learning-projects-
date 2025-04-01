from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, desc, when,mean , stddev
from pyspark.sql.types import *
from pyspark.sql.window import Window
from math import floor, ceil


spark = SparkSession.builder.appName('statics').getOrCreate()

df = spark.read.csv(
        '/content/drive/MyDrive/countries_population.csv',
        header = True,
        inferSchema = True
    )
df.printSchema()
mean_p = df.agg(mean(col('Population Growth')).alias('means')).first()[0]
mean_g = df.agg(mean(col('Growth Rate (%)')).alias('means')).first()[0]

# '''
df = df.fillna(mean_p , subset = ['Population Growth'] )
df = df.fillna(mean_g, subset = ['Growth Rate (%)'])
print(df.show())
# '''
# df = df.withColumn('Population Growth', when(
#         col('Population Growth').isNull() , mean_p
#         ).otherwise( col('Population Growth') ) )

# null_values  = df.select(sum(when(col('Population Growth').isNull(), 1).otherwise(0)).alias('null_col'))
# print(f' null values in that : {null_values.collect()[0][0]}')
types = [IntegerType(), DoubleType(), LongType(), FloatType(),DecimalType()]

schema = df.schema

name_dtype = [(feilds.name,feilds.dataType) for feilds in schema.fields]
inter_col = [name  for name, dtype in name_dtype if dtype in types if name != 'Year']
col_means = []
col_std = []
col_med = []
for col_ in inter_col:
    mean_values = float(df.select(mean(col_)).collect()[0][0])
    std_values = float(df.select(stddev(col_)).collect()[0][0])

    col_means.append((col_,mean_values))
    col_std.append((col_,std_values))

    N = df.select(col_).count()
    main_value = N / 2
    first_index = floor(main_value)
    second_index = ceil(main_value)

    y = df.select(col_)
    windows = Window().orderBy(y[col_])
    y = y.withColumn('row_number',row_number().over(windows))
    y= y.filter((y['row_number'] == first_index) | (y['row_number'] == second_index))
    values  = 0
    for i in range(y.count()):
        values += y.collect()[i][0]
    median = values / y.count()
    col_med.append((col_,median))
print( ' outliers points')
mean_ = float(df.select(mean('Population')).collect()[0][0])
stddev_ = float(df.select(stddev('Population')).collect()[0][0])

outliers = df.filter(df['Population'] > (mean_ + stddev_ * 3))
print(outliers.show())

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import *

schema = df.schema.fields

dtypes = [LongType(), IntegerType(), DoubleType(), FloatType()]
columns_list = [filed.name  for filed in schema if filed.dataType in dtypes]
df_columns = df.columns
cov_matrix = np.zeros((len(df_columns), len(df_columns)))
corr_matrix = np.zeros((len(df_columns), len(df_columns)))

for i in range(len(df_columns)):
    for j in range(len(df_columns)):
        data = df
        x = df_columns[i]
        y = df_columns[j]
        if x in columns_list:
            if y in columns_list:
                
                x_mean = data.select(F.mean(x).alias('first_features')).first()[0]
                y_mean = data.select(F.mean(y).alias('second_features')).first()[0]
                x_std = data.select(F.stddev(x).alias('std_x')).first()[0]
                y_std = data.select(F.stddev(y).alias('std_y')).first()[0] 

                corr = df.select(F.sum((col(x) - x_mean) * (col(y) - y_mean))).first()[0]
                covex = df.select(x).count() - 1

                v = corr / covex
                if v > 0 :
                    cov_matrix[i,j] = 1.0
                elif v == 0 :
                    cov_matrix[i,j] = 0.0
                else:
                    cov_matrix[i,j] = -1.0
                
                corr_matrix[i,j] = round((v / ( x_std * y_std)) ,4) 

print(cov_matrix)
print(corr_matrix)

import pandas as pd 
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/regression_dataset.csv')
unique_values = {values : index for index , values in enumerate(np.unique(df['Feature4']))}
df['Feature4'] = df['Feature4'].apply(lambda x : unique_values[x])

def compute_loss_(y_true, y_pred): 
    return np.mean(np.square(y_true - y_pred)) 

y_true = np.array([3.0, 5.0, 2.5, 7.0])  
y_pred = np.array([0.1, 0.2, 2.0, 7.8])  
h = 0.00001 
gradient = ((compute_loss_(y_true, y_pred) + h) - compute_loss_(y_true,y_pred)) / h 

# y_true = df['Target']

# that is for if you working with 1-1  
def coff(x): 
    y = df['Target']
    x_mean = x.mean() 
    y_mean = y.mean() 

    diffx = x - x_mean 
    diffy = y - y_mean 
    upper = np.sum(diffx * diffy)
    cov = upper / (len(x) - 1 )
    var = np.sum(np.square(x - x_mean)) / (len(x) - 1)  
    temp = x.std() 
    return cov / var  

# but before that you need to add a ones colum over here 

df['beta'] = 1 
df = df[['beta'] + [col_ for col_ in df.columns if col_ != 'beta']]
x = df.drop('Target' , axis = 1)
y = df['Target']

# computeing inter. coff  ( x.T @ x ).inv() @ (x.T @ y) 
beta_coff = np.linalg.inv((x.T @ x )) @ (x.T @ y )

def compute_loss__(X, y, beta):
    residuals = y - X @ beta
    return np.mean(residuals ** 2)

# Gradient Descent (optimized)
def gradient_descent__(X, y, alpha=0.01, epochs=1000):
    n, p = X.shape
    beta = np.zeros(p)
    losses = []
    
    for epoch in range(epochs):
        residuals = y - X @ beta
        gradient = -2/n * X.T @ residuals  # Analytical gradient
        beta -= alpha * gradient
        
        # Track loss every 100 epochs
        if epoch % 100 == 0:
            loss = compute_loss__(X, y, beta)
            losses.append(loss)
            print(f'Epoch {epoch}: Loss = {loss:.4f}')
    
    return beta, losses

# Run gradient descent
beta_optimized, loss_history = gradient_descent__(x, y, alpha=0.01, epochs=1000)
print("Optimized Coefficients:", beta_optimized)


