from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import json

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

print("Loading test data...")
test_data = spark.read.parquet("/project/data/test")

print("Loading model...")
model = PipelineModel.load("/project/models/")

print("Running predictions...")
predictions = model.transform(test_data)


evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auroc = evaluator.evaluate(predictions)

print(f"Model AUROC: {auroc:.4f}")

# 🚨 Critical MLOps Step: Evaluation Gate
THRESHOLD = 0.5

metrics = {
    "auroc": auroc,
    "status": "passed" if auroc >= THRESHOLD else "failed"
}

with open("/project/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to metrics.json")

if auroc < THRESHOLD:
    raise Exception(f"Model performance too low! AUROC={auroc}")

print("Model passed evaluation!")

spark.stop()