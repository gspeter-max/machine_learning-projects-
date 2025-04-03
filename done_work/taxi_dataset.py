from pyspark.sql import  SparkSession
spark = SparkSession.builder.appName('genius').getOrCreate()

df = spark.read.csv('/content/drive/MyDrive/nyc_taxi_messy.csv', header = True, inferSchema= True )
df.show()


'''
1. show null values in ( df.isnull().sum()) in pandas like
-- pandas
df.isnull().sum()

-- spark
but that is slow

import numpy as np
null_dict = {}
for col in df.columns :
    i = 0
    for values in df.select(col).collect():
        if values[0] == None or values[0] == np.nan:
            i += 1
    null_dict[col] = i
import pandas as pd
null_data = pd.DataFrame(null_dict, index = [1])
null_data = null_data.transpose()
null_data

'''

from pyspark.sql.functions import count, col
a = []
for c in df.columns:
	t = (count("*") - count(col(c))).alias(f'{c}')
	a.append(t)

values = df.agg(*a)
values.show()
import pyspark.sql.functions  as f

x = set(v.dataType for  v in df.schema)
from pyspark.sql.types import *
a = [DoubleType(), IntegerType()]

for _Col_ in [feilds.name for feilds in df.schema if feilds.dataType in a ]:
    print(_Col_)
    mean = df.select(f.mean(f.col(_Col_))).collect()[0]
    df = df.fillna(value = float(mean[0]), subset=[_Col_])

from pyspark.sql.window import Window
from pyspark.sql.functions import last , when , count , lit, mode, col

# back_wind = Window().orderBy('extra_notes').rowsBetween(Window.unboundedPreceding, 0)
# forward_wind  = Window().orderBy('extra_notes').rowsBetween(0,1 )

# df = df.withColumn('extra_notes', when(col('extra_notes').isNull() , last(col('extra_notes'),ignorenulls= True).over(back_wind)).otherwise(col('extra_notes')))

a = []
total_sum = count(lit(1))
for c in df.columns:
    b  = (total_sum - count(col(c))).alias(c)
    a.append(b)

x = df.agg(*a)
x.show()

co = col('extra_notes')
x = df.filter(co.isNotNull()).groupby(co).agg(count(lit(co)).alias('frequency')).collect()[0][0]
df = df.withColumn('extra_notes', when(co.isNull(),str(x)).otherwise(co))
df = df.dropDuplicates()


print(f' initial particle : {df.rdd.getNumPartitions()}')
df = df.repartition(8)
print(f'final particle : {df.rdd.getNumPartitions()}')

# df.write.option('compression', 'snappy').parquet('/content/drive/MyDrive/nyc_taxi_messy.parquet')
