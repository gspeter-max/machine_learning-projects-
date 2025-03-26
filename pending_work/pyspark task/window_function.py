from pyspark.sql import SparkSession 

spark = SparkSession.builder.appName('sql_window').getOrCreate() 

df = spark.read.csv('/workspaces/machine_learning-projects-/pending_work/pyspark task/small_media_data.csv', header = True,
                    inferSchema = True) 


'''
ROW_NUMBER, RANK, LEAD, LAG
'''

# sql 

# df.createOrReplaceTempView('media_table') 

# df = spark.sql(
#     '''
#         with temp_temp as (
#             select 
#                 row_number() over( partition by media_type order by likes desc ) as ranking, 
#                 rank() over(partition by media_type order by likes desc) as rank_ranking , 
#                 lead(likes) over ( order by timestamp ) as lead_function , 
#                 lag(likes) over( order by timestamp) as leg_function
#             from media_table 
#         )
        
#         select * from temp_temp ; 
    
#     '''
# ).show() 

from pyspark.sql  import window 

windows = window.orderBy(df.likes.desc()) 
df = df.withColumn('ranking', row_number().over(windows) ) 
print(df.show()) 
