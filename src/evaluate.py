from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder \
    .appName("Airline_Project_Unit4_Pro") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.ansi.enabled", "false") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Loading test data...")
test_data = spark.read.parquet("data/test")

print("Loading model...")
model = PipelineModel.load("models/model")

print("Running predictions...")
predictions = model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auroc = evaluator.evaluate(predictions)

print(f"Model AUROC: {auroc:.4f}")

# 🚨 Critical MLOps Step: Evaluation Gate
THRESHOLD = 0.7

if auroc < THRESHOLD:
    raise Exception(f"Model performance too low! AUROC={auroc}")

print("Model passed evaluation!")

spark.stop()