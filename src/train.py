from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# 1. SETUP: Connect to the Dockerized Spark Master
spark = SparkSession.builder \
    .appName("Airline_Project_Unit4_Pro") \
    .master("spark://spark-master:7077") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.ansi.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Loading processed data...")
df = spark.read.parquet("/project/data/processed")

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Save test set for evaluation
test_data.write.mode("overwrite").parquet("/project/data/test")

# Feature Engineering
carrier_indexer = StringIndexer(inputCol="CarrierCode", outputCol="CarrierIndex", handleInvalid="keep")
origin_indexer = StringIndexer(inputCol="Origin", outputCol="OriginIndex", handleInvalid="keep")

encoder = OneHotEncoder(
    inputCols=["CarrierIndex", "OriginIndex"],
    outputCols=["CarrierVec", "OriginVec"],
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=["Month", "DayofMonth", "Distance", "CRSDepTime", "CarrierVec", "OriginVec"],
    outputCol="features",
    handleInvalid="skip"
)

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

pipeline = Pipeline(stages=[carrier_indexer, origin_indexer, encoder, assembler, lr])

print("Training model...")
model = pipeline.fit(train_data)

print("Saving model...")
model.write().overwrite().save("/project/models")

spark.stop()