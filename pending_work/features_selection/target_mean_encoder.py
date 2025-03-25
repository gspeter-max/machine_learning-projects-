
from pyspark.sql import DataFrame 
import pandas as pd 
from typing import Union
import numpy as np
from pyspark.sql import SparkSession 

spark = SparkSession.builder.appName('featurs_selection').getOrCreate() 
df = spark.read.csv('/workspaces/machine_learning-projects-/pending_work/features_selection/google_scale_synthetic_dataset.csv',
                    header = True, inferSchema = True 
) 

'''
Pandas loads everything into RAM. If your dataset is huge (e.g., 100GB), but your RAM is only 16GB,
 it will crash or slow down drastically.

'''
df_pandas = df.toPandas() 

class target_encoder: 
    
    def __init__(self,strategy, data : Union[pd.DataFrame, DataFrame], categorical_features : Union[list,str, np.ndarray],target_feature = None):  
        
        self.strategy = strategy 
        self.target_feature = target_feature 
        
        if isinstance(data , DataFrame): 
            data = data.toPandas() 
        
        self.data = data 
        self.categorical_features = categorical_features 

    def call(self): 
        if isinstance( self.categorical_features, str): 
            if self.strategy.lower() == 'auto': 
                
                self.data[self.categorical_features] = self.data[self.categorical_features].replace([values for index, values in enumerate(self.data[self.categorical_features].unique())] ,
                                                                        [index + 1 for index, values in enumerate(self.data[self.categorical_features].unique())]
                                                                                                    ) 
                return self.data

            if self.strategy.lower() == 'mean': 
                if self.target_feature is None: 
                    raise ValueError('None is not allow ') 
                
                groupby_data = self.data.groupby(self.categorical_features)[self.target_feature].agg('mean') 
                groupby_data = groupby_data.reset_index()
                groupby_data.columns = [self.categorical_features, self.categorical_features + 'encoded'] 
                df = pd.merge(
                    self.data , 
                    groupby_data, 
                    on = self.categorical_features, 
                    how = 'left' 
                )
                df = df.drop(self.categorical_features, axis = 1 ) 
                return df 

            else: 
                print(' that part  is not add') 
                raise ValueError(' that is not include in this') 
        
        else: 
            for categorical_column  in self.categorical_features:
                if self.strategy.lower() == 'auto': 
        
                    self.data[categorical_column] = self.data[categorical_column].replace([values for index, values in enumerate(self.data[categorical_column].unique())] ,
                                                                        [index + 1 for index, values in enumerate(self.data[categorical_column].unique())]
                                                                                                    ) 

                if  self.strategy.lower() == 'mean': 
                    if self.target_feature is None: 
                        raise ValueError('None is not allow ') 
                    
                    groupby_data = self.data.groupby(categorical_column)[self.target_feature].agg('mean') 
                    groupby_data = groupby_data.reset_index()
                    groupby_data.columns = [categorical_column, categorical_column + 'encoded'] 
                    self.data = pd.merge(
                        self.data , 
                        groupby_data, 
                        on = categorical_column, 
                        how = 'left' 
                    )
                    self.data = self.data.drop(categorical_column, axis = 1 ) 
                
                else:
                    raise ValueError('that strategy is not included') 
            
            return self.data

cat_column = ['cat_feature_0', 'cat_feature_1',
       'cat_feature_2', 'cat_feature_3', 'cat_feature_4', 'cat_feature_5',
       'cat_feature_6', 'cat_feature_7', 'cat_feature_8', 'cat_feature_9']

encoder = target_encoder(strategy = 'mean',data = df_pandas ,
                         categorical_features = cat_column ,target_feature = 'target')

df_ = encoder.call()