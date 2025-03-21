from pyspark.sql.functions import rand, avg , stddev , when, least,sqrt,col 
from functools import reduce 
class lloyd_algo:

    def __init__(self, k = 3 , max_iteration = 10 , input_col_index = [0,1,2]):
        
        self.k = k
        self.max_iteration = max_iteration
        self.input_col_index = input_col_index
        self.temp_dict = {}

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
            scaled_features.append(f.col(f'{featues__}_scaled'))


        self.df = self.df.withColumn(combine_feature_name,reduce(lambda x ,y : x + y , scaled_features))
        self.temp_dict['combine_features'] = combine_feature_name

#       input ( centroid_values , column )
        # that main thing is here i think 
    def compute_distance(self, centroid_value , column, make_column_name):
        
        self.df = self.df.withColumn(make_column_name, (col(column) - centroid_value)**0.5)
    
    
    # call  that that give you a distance intial 

    def call(self,inside_enumerate_functions, last = False):
        distance = [] 
        for index,centroid in enumerate(inside_enumerate_functions):
            
            name = f'centroid_{index + 1}'
            self.temp_dict[name] = centroid[0]
            COL_D = f'cen_{index + 1}_distance'
            self.temp_dict[f'compute_distance_column{index + 1}'] = COL_D
            self.compute_distance(self.temp_dict[name], self.temp_dict['combine_features'], COL_D)

            distance.append(col(COL_D))
        self.temp_dict['distance'] = distance
    
        # modified that remove that diff and ujisng i think for
    def make_ranking(self, distance = None,n = 0):
        
        if distance is None : 
            raise ValueError('distance is not be error')
        
        trakers = None 
        
        for i in range(len(distance)): 

            initial_condition = when(distance[i] ==  least(*distance), i + 1)
            trakers = initial_condition if trakers is None else initial_condition.when(distance[i] == least(*distance),i + 1 )
        
        self.df = self.df.withColumn(f'cluster_{n}', trakers)
       

    def recompute_means(self, features : list = None): 
        if features is None:
            features = [f'Feature{i + 1}_scaled' for i in  range(self.k)]

        if self.max_iteration is None : 
            raise ValueError( 'None is not allow')


        for i in range(self.max_iteration):

            if i == 0 : 
                random_row = self.df.select(self.temp_dict['combine_features']).orderBy(rand()).limit(self.k).collect() #  DIFF
                inside_enumerate_function  = [iter_df for iter_df in random_row]
                print(inside_enumerate_function)
                
                    
            else: 
            # 1.recompute mean as a new centroid and copute for each k ( not 3 )

                # and another thing is here 
# may be that computes for cluster right but that cluster column have only 3 centroid access 

                mean_df = self.df.groupby(f'cluster').agg(*[
                    avg(col).alias(f'{col}_mean') for col in features 
                ])

                mean_features = [f'Feature{i + 1}_scaled_mean' for i in range(self.k)]

                mean_df = mean_df.withColumn('sums', reduce(lambda x, y : x + y , (abs(col(c)) for c in mean_features)))
                inside_enumerate_function = mean_df.select('sums').collect()
    
            
            self.call(inside_enumerate_functions = inside_enumerate_function,)
            self.make_ranking(distance= self.temp_dict['distance'], n = i)

            # breaking statement for using subtraction of that cluster_n
            print(f"=============================={i}================================================")
            if i != 0 : 

                variable = 0 
                if self.df.select(sum(abs(col(f'cluster_{i-1}') - col(f'cluster_{i}')))).collect()[0][0] <= 10 :
                    
                    print('cluster is found check out your dataframe uisng {YOUR_CLASS}.df')
                    break






y = lloyd_algo()
y.load_data()
y.recompute_means()
