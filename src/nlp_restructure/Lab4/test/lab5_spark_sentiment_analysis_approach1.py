import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr, size
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as F

# --- Configurable parameters ---
MIN_FREQ = 5
MAX_FREQ = 1000
MIN_SENT_LEN = 3
NUM_FEATURES = 3000

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisApproach1").getOrCreate()

BASE_DIR = Path(__file__).parents[1].resolve()

# 2. Load Data
data_path = str(BASE_DIR / "data" / "sentiments.csv")
df = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Initial samples: {df.count()}")

# 3. Noise Filtering
clean_text = regexp_replace(lower(col("text")), r"https?://\S+|www\.\S+", "")
clean_text = regexp_replace(clean_text, r"<.*?>", "")
clean_text = regexp_replace(clean_text, r"[^a-zA-Z\s]", "")
df = df.withColumn("clean_text", clean_text)

# 4. Prepare label
df = df.withColumn("label", expr("try_cast(sentiment as int)"))
df = df.dropna(subset=["label", "clean_text"])
df = df.filter(col("label").isin(-1, 1))
df = df.withColumn("label", expr("CASE WHEN label = -1 THEN 0 ELSE 1 END"))

# 5. Tokenize and remove stopwords
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = tokenizer.transform(df)
df = stopwordsRemover.transform(df)

# 6. Vocabulary Reduction
word_df = df.select(F.explode(col("filtered_words")).alias("word"))
word_freq = word_df.groupBy("word").count()
freq_words_df = word_freq.filter((col("count") >= MIN_FREQ) & (col("count") <= MAX_FREQ)).select("word")

freq_words_list = [row["word"] for row in freq_words_df.collect()]

freq_words_col = F.array([F.lit(w) for w in freq_words_list])

df = df.withColumn("filtered_words_final", 
                   F.array_intersect(col("filtered_words"), freq_words_col))
print(f"After reduce vocabulary: {df.count()}")

# 7. Feature Extraction
# TF-IDF
hashingTF = HashingTF(inputCol="filtered_words_final", outputCol="raw_features", numFeatures=NUM_FEATURES)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

# 8. Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 9. Build pipeline
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="tfidf_features", labelCol="label")
pipeline = Pipeline(stages=[hashingTF, idf, lr])

# 10. Training and evaluation
train_start = time.time()
model = pipeline.fit(train_df)
train_time = time.time() - train_start

eval_start = time.time()
predictions = model.transform(test_df)
eval_time = time.time() - eval_start

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1_score = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)

# Save result
result_dir = BASE_DIR / "results"
result_dir.mkdir(parents=True, exist_ok=True)
output_path = result_dir / "lab5_spark_sentiment_approach1_results.txt"
with open(output_path, 'w', encoding='utf-8') as f:
	f.write(f"Model training time: {train_time:.3f} seconds\n")
	f.write(f"Model evaluation time: {eval_time:.3f} seconds\n")
	f.write(f"Test Accuracy: {accuracy:.3f}\n")
	f.write(f"Test F1 Score: {f1_score:.3f}\n")

# Print result
print(f"Model training time: {train_time:.3f} seconds")
print(f"Model evaluation time: {eval_time:.3f} seconds")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Test F1 Score: {f1_score:.3f}")

spark.stop()
