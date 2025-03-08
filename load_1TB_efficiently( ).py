from pyspark.sql import SparkSession
from pyspark.sql.functions import sum
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

# Initialize Spark Session
spark = SparkSession.builder \
    .appName('ai_data_processing') \
    .config('spark.sql.shuffle.partitions', 500) \
    .getOrCreate()

# Load CSV with correct syntax
df = spark.read.csv('/content/linearly_separable_classification.csv', header=True, inferSchema=True)

# Repartition the dataframe to optimize parallelism
df = df.repartition(100)

# Convert to Parquet (Uncomment if needed)
# df.write.mode('overwrite').parquet('/content/linearly_separable_classification.parquet')

# Read Parquet file
df_parquet = spark.read.parquet('/content/linearly_separable_classification.parquet')

# Ensure partitioning is correct
df_parquet = df_parquet.repartition(1000)

# Fill NaN values
df_parquet = df_parquet.fillna({'colname': sum(df_parquet['colname'])})

# Feature Engineering
feature_cols = [col for col in df_parquet.columns if col != 'label']  # Exclude label column
vector = VectorAssembler(inputCols=feature_cols, outputCol='vector_features')
df_parquet = vector.transform(df_parquet)

# Logistic Regression Model
lr = LogisticRegression(featuresCol='vector_features', labelCol='label')
model = lr.fit(df_parquet)

# Show output
df_parquet.show(5)
