from pyspark.sql import SparkSession 

spark = SparkSession.builder.appName('sql_window').getOrCreate() 

df = spark.read.csv('/workspaces/machine_learning-projects-/pending_work/pyspark task/small_media_data.csv', header = True,
                    inferSchema = True) 


'''
ROW_NUMBER, RANK, LEAD, LAG
'''

# sql 

df.createOrReplaceTempView('media_table') 

spark.sql(
    '''
        with temp_temp as (
            select 
                row_number() over( partition by media_type order by likes desc ) as ranking, 
                rank() over(partition by media_type order by likes desc) as rank_ranking , 
                lead(likes) over ( order by timestamp ) as lead_function , 
                lag(likes) over( order by timestamp) as leg_function
            from media_table 
        )
        
        select * from temp_temp ; 
    
    '''
).show() 

from pyspark.sql.window import Window 
from pyspark.sql.functions import  row_number 

windows = Window.partitionBy(df.user_id).orderBy(df.likes.desc()) 
df = df.withColumn('ranking', row_number().over(windows) ) 


from pyspark.sql.functions import rank , col 
from pyspark.sql.window import Window 


windows = Window.partitionBy(col('user_id')).orderBy(col('likes').desc())  
df = df.withColumn('rank_ranking', rank().over(windows)) 
df.show()

from pyspark.sql.functions import  lead , lag, col 
from pyspark.sql.window import Window 

windows = Window.orderBy(col('likes').desc()) 

df = df.withColumn('lead', lead(col('likes')).over(windows)) 
df = df.withColumn('leg', lag(col('likes')).over(windows)) 

df.show()
from pyspark.sql.functions import  lead , lag, col 
from pyspark.sql.window import Window 

windows = Window.orderBy(col('likes').desc()) 

df = df.withColumn('lead', lead(col('likes')).over(windows)) 
df = df.withColumn('leg', lag(col('likes')).over(windows)) 

df.show()