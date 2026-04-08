from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, concat_ws, round, when, broadcast

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

# 2. EXTRACT: Load Both Datasets
print("Loading data...")
flights_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/project/data/airline.csv.shuffle")

carriers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/project/data/carriers.csv") \
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
# print("Generating Aggregated CSVs for R and PowerBI...")
# daily_stats = clean_df.groupBy("FlightDate").agg(
#     round(avg("DepDelay"), 2).alias("AvgDepDelay"),
#     round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
#     count("*").alias("TotalFlights")
# ).orderBy("FlightDate")

# carrier_stats = clean_df.groupBy("CarrierName").agg(
#     round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
#     count("*").alias("TotalFlights"),
#     round((count(when(col("Cancelled") == 1, True)) / count("*")) * 100, 2).alias("CancelRate")
# ).orderBy(col("TotalFlights").desc())

# airport_stats = clean_df.groupBy("Origin").agg(
#     count("*").alias("TotalDepartures"),
#     round(avg("DepDelay"), 2).alias("AvgDepDelay")
# ).orderBy(col("TotalDepartures").desc())

# 5. LOAD: Save Files
# output_path = "project_outputs"
# daily_stats.coalesce(1).write.csv(f"{output_path}/daily_trends", header=True, mode="overwrite")
# carrier_stats.coalesce(1).write.csv(f"{output_path}/carrier_stats_named", header=True, mode="overwrite")
# airport_stats.coalesce(1).write.csv(f"{output_path}/airport_stats", header=True, mode="overwrite")
# print("Aggregations Generated (Save bypassed).")

# 6. DISTRIBUTED MACHINE LEARNING (With Categorical Encoding & Pipeline)
print("\n--- Starting Distributed Machine Learning Pipeline ---")

# Pulling in the new categorical columns
ml_df = clean_df.select("Month", "DayofMonth", "Distance", "CRSDepTime", "CarrierCode", "Origin", "ArrDelay") \
    .sample(withReplacement=False, fraction=0.1, seed=42) \
    .withColumn("label", when(col("ArrDelay") > 15, 1.0).otherwise(0.0))

print("Saving processed data...")

ml_df.write.mode("overwrite").parquet("/project/data/processed")

spark.stop()