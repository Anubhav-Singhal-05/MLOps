from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, concat_ws, round, when, broadcast
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow

# 1. SETUP: Initialize Spark with ANSI disabled
spark = SparkSession.builder \
    .appName("Airline_Project_Unit4_Pro") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.ansi.enabled", "false") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. EXTRACT: Load Both Datasets
print("Loading data...")
flights_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("airline.csv.shuffle")

carriers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("carriers.csv") \
    .withColumnRenamed("Code", "CarrierCode") \
    .withColumnRenamed("Description", "CarrierName")

# 3. TRANSFORM: Join & Clean
print("Joining Flights with Carrier Names and Cleaning...")
joined_df = flights_df.join(broadcast(carriers_df), flights_df.UniqueCarrier == carriers_df.CarrierCode, "left")

clean_df = joined_df.withColumn(
    "FlightDate",
    to_date(concat_ws("-", col("Year"), col("Month"), col("DayofMonth")))
).withColumn(
    "ArrDelay", col("ArrDelay").cast("float")
).withColumn(
    "DepDelay", col("DepDelay").cast("float")
).withColumn(
    "Distance", col("Distance").cast("float")
).withColumn(
    "CRSDepTime", col("CRSDepTime").cast("float")
).withColumn(
    "Month", col("Month").cast("integer")
).withColumn(
    "DayofMonth", col("DayofMonth").cast("integer")
).withColumn(
    "Cancelled", col("Cancelled").cast("integer")
).na.drop(subset=["ArrDelay", "Month", "DayofMonth", "Distance", "CRSDepTime", "CarrierCode", "Origin"])

# 4. AGGREGATE: Create Outputs
print("Generating Aggregated CSVs for R and PowerBI...")
daily_stats = clean_df.groupBy("FlightDate").agg(
    round(avg("DepDelay"), 2).alias("AvgDepDelay"),
    round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
    count("*").alias("TotalFlights")
).orderBy("FlightDate")

carrier_stats = clean_df.groupBy("CarrierName").agg(
    round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
    count("*").alias("TotalFlights"),
    round((count(when(col("Cancelled") == 1, True)) / count("*")) * 100, 2).alias("CancelRate")
).orderBy(col("TotalFlights").desc())

airport_stats = clean_df.groupBy("Origin").agg(
    count("*").alias("TotalDepartures"),
    round(avg("DepDelay"), 2).alias("AvgDepDelay")
).orderBy(col("TotalDepartures").desc())

# 5. LOAD: Save Files
output_path = "project_outputs"
# daily_stats.coalesce(1).write.csv(f"{output_path}/daily_trends", header=True, mode="overwrite")
# carrier_stats.coalesce(1).write.csv(f"{output_path}/carrier_stats_named", header=True, mode="overwrite")
# airport_stats.coalesce(1).write.csv(f"{output_path}/airport_stats", header=True, mode="overwrite")
print("Aggregations Generated (Save bypassed).")

# 6. DISTRIBUTED MACHINE LEARNING (With Categorical Encoding & Pipeline)
print("\n--- Starting Distributed Machine Learning Pipeline ---")

# Pulling in the new categorical columns
ml_df = clean_df.select("Month", "DayofMonth", "Distance", "CRSDepTime", "CarrierCode", "Origin", "ArrDelay") \
    .sample(withReplacement=False, fraction=0.1, seed=42) \
    .withColumn("label", when(col("ArrDelay") > 15, 1.0).otherwise(0.0))

print("Splitting data into Train and Test sets...")
# CRITICAL: We split BEFORE building the pipeline to prevent target leakage
train_data, test_data = ml_df.randomSplit([0.8, 0.2], seed=42)

# Step A: String Indexers (String -> Number)
carrier_indexer = StringIndexer(inputCol="CarrierCode", outputCol="CarrierIndex", handleInvalid="keep")
origin_indexer = StringIndexer(inputCol="Origin", outputCol="OriginIndex", handleInvalid="keep")

# Step B: One-Hot Encoder (Number -> Binary Vector)
encoder = OneHotEncoder(inputCols=["CarrierIndex", "OriginIndex"], outputCols=["CarrierVec", "OriginVec"], handleInvalid="keep")

# Step C: Assemble all features
assembler = VectorAssembler(
    inputCols=["Month", "DayofMonth", "Distance", "CRSDepTime", "CarrierVec", "OriginVec"],
    outputCol="features",
    handleInvalid="skip"
)

# Step D: Logistic Regression Estimator
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# Orchestrate the Pipeline
print("Training Pipeline (Index -> Encode -> Assemble -> Train)...")
pipeline = Pipeline(stages=[carrier_indexer, origin_indexer, encoder, assembler, lr])

# Fit the pipeline on training data
pipeline_model = pipeline.fit(train_data)

# Important: Save the entire pipeline model (including encoders) for consistent transformations during evaluation and future predictions
pipeline_model.write().overwrite().save("models/model")

print("Evaluating Model...")
# Transform test data through the exact same pipeline
predictions = pipeline_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auroc = evaluator.evaluate(predictions)

mlflow.start_run()
mlflow.log_metric("auroc", auroc)
mlflow.end_run()

print(f"\n>> ML PIPELINE COMPLETE <<")
print(f">> Model Accuracy (Area Under ROC): {auroc:.4f}")
print(f">> Categoricals encoded via Pipeline architecture.\n")

spark.stop()