from pyspark.sql.functions import rand, avg , stddev , when, least
from functools import reduce 
class lloyd_algo:

    def __init__(self, k = 3 , max_iteration = 2 , input_col_index = [0,1,2]):
        
        self.k = k
        self.max_iteration = max_iteration
        self.input_col_index = input_col_index

    def load_data(self, path = '/content/drive/MyDrive/kmeans_data.csv',combine_feature_name = 'features'):

        from pyspark.sql import SparkSession
        from functools import reduce
        from pyspark.sql import functions as f

        
        
        spark = SparkSession.builder.appName('k-mean').getOrCreate()
        self.df  = spark.read.csv(path, header = True, inferSchema = True)

        
        
        if self.input_col_index is None:
            raise IndexError(' input_col_index  not  be None')

        
        DF_COLUMNS = [self.df.columns[i] for i in self.input_col_index]
        
        scaled_features = [] 
        for featues__ in DF_COLUMNS: 
            
            mean = self.df.select(avg(col(featues__))).collect()[0][0]
            stddev_ = self.df.select(stddev(col(featues__))).collect()[0][0]
            self.df = self.df.withColumn(f'{featues__}_scaled', (col(featues__) - mean)/ stddev_)
            scaled_features.append(f'{featues__}_scaled')



        columns_list = [f.col(i) for i in scaled_features]


        self.df = self.df.withColumn(combine_feature_name,reduce(lambda x ,y : x + y , columns_list))
        self.combine_feature = combine_feature_name

#       input ( centroid_values , column )

    def compute_distance(self, centroid_value , column, column_name):
        
        from pyspark.sql.functions import sqrt,col

        self.df = self.df.withColumn(column_name, sqrt((col(column) - centroid_value)**2))
    
    
    # call  that that give you a distance intial 

    def call(self):

        from pyspark.sql.functions import when, mean, col, least

        random_row = self.df.orderBy(rand()).limit(self.k).collect()
        row_list = self.df.select(self.combine_feature).collect()
        
        
         # cluster values in list
        random_values = [iter_df[self.combine_feature] for iter_df in random_row]
        print(f'random_values {random_values}')
        distance = [] 

        for index,centroid in enumerate(random_values):

            name = f'centroid_{index + 1}'
            setattr(self,name , centroid)

            COL_D = f'cen_{index + 1}_distance'
            self.compute_distance(getattr(self,name), self.combine_feature, COL_D)

            distance.append(col(COL_D))
        
        return distance 
    

    def make_ranking(self,diff_1 = None, diff_2 = None,  diff_3 = None , distance = None,n = 0):
        if distance is None : 
            raise ValueError('distance is not be error')

        self.df = self.df.withColumn(f'cluster_{n}', 
            when(col(diff_1) == least(*distance), 1)
            .when(col(diff_2) == least(*distance), 2)
            .when(col(diff_3) == least(*distance), 3)
)


    def recompute_means(self, features : list = ['Feature1_scaled', 'Feature2_scaled','Feature3_scaled']): 
        
        from pyspark.sql.functions import avg 
        # inital random clusters 
        distance = self.call() 
        self.make_ranking('cen_1_distance','cen_2_distance','cen_3_distance', distance)


        for i in range(self.max_iteration):
            i = 1 
            # 3. pass a condition 3nd centroid is same as a starting
            if i == 3 : 
                if getattr(self,centroid_1) == getattr(self,_3nd_mean_1)  and getattr(self,centroid_2) == getattr(self,_3nd_mean_2) and getattr(self,centroid_3) == getattr(self,_3nd_mean_3): 
                    return self.df.select('cluster')
            
            # 1.recompute mean as a new centroid and copute for each k ( not 3 )

            mean_df = self.df.groupby('cluster').agg(*[
                avg(col).alias(f'{col}_mean') for col in features 
            ])
            mean_df.show()
            mean_features = ['Feature1_scaled_mean', 'Feature2_scaled_mean','Feature3_scaled_mean']

            mean_df = mean_df.withColumn('sums', reduce(lambda x, y : x + y , (abs(col(c)) for c in mean_features)))

            '''
                        +-------+-------------------+------------------+------------------+------------------+
                        |Cluster|      Feature1_mean|     Feature2_mean|     Feature3_mean|              temp|
                        +-------+-------------------+------------------+------------------+------------------+
                        |      1| 1.9639439578407425|-6.820876151420051|-6.853002497387031|15.637822606647825|
                        |      2| -8.970689822608819| 7.204271488878844|2.0917432103080014|18.266704521795667|
                        |      0|-2.5292536495167792| 9.167289977967302| 4.759840624025556|16.456384251509636|
                        +-------+-------------------+------------------+------------------+------------------+
            

            
            '''
            distance = [] 
            for index, value in enumerate(mean_df.select('sums').collect()): 
                value = value[0]
                index = index + 1 
    
                setattr(self, f'{i}nd_mean_{index}',value)
                column_names = f'{i}features_mean{index}'
                name = f'{i}nd_mean_{index}'
                self.compute_distance(getattr(self,name), self.combine_feature,column_names)
                
                distance.append(col(column_names))
            
            self.make_ranking(diff_1 = f'{i}features_mean1',diff_2 = f'{i}features_mean2',diff_3 = f'{i}features_mean3',distance= distance, n = i)

            if self.df.select(f'cluster_{i}') == self.df.select(f'cluster_{i-1}'): 
                break 
    
y = lloyd_algo()
y.load_data()
y.recompute_means()
