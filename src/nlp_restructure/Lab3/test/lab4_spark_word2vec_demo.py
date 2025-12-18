import time
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, size
from pyspark.ml.feature import Word2Vec, StopWordsRemover, Tokenizer
import pyspark.sql.functions as F

root_dir = Path(__file__).parents[1]


def create_spark_session(output_file):
    spark = (
        SparkSession.builder.appName("Lab4_PySpark_Word2Vec")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    output_file.write("Spark initialized\n")
    return spark


def load_and_preprocess_data(spark, data_path: Path, output_file, sample_fraction=1.0):
    output_file.write("Loading data\n")

    df = spark.read.json(str(data_path)).sample(fraction=sample_fraction, seed=42)

    cleaned_df = df.select(
        regexp_replace(
            regexp_replace(lower(col("text")), r"[^\w\s]", ""), r"\s+", " "
        ).alias("cleaned_text")
    ).filter(F.length(col("cleaned_text")) > 50)

    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="raw_words")
    stop_remover = StopWordsRemover(inputCol="raw_words", outputCol="words")

    processed_df = stop_remover.transform(tokenizer.transform(cleaned_df))
    final_df = processed_df.filter(size(col("words")) >= 3).select("words")

    count = final_df.count()
    output_file.write(f"Processed {count:,} documents\n")
    return final_df


def train_word2vec_model(df, output_file, vector_size=100, min_count=3, max_iter=5):
    output_file.write("Training Word2Vec model\n")

    training_start_time = time.time()

    word2vec = Word2Vec(
        inputCol="words",
        outputCol="features",
        vectorSize=vector_size,
        minCount=min_count,
        maxIter=max_iter,
        seed=42,
    )

    model = word2vec.fit(df)

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time

    output_file.write(
        f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)\n"
    )

    return model


def demonstrate_model_usage(model, output_file, test_word="computer", top_n=5):
    vectors = model.getVectors()
    vocab_size = vectors.count()

    output_file.write(f"\nVocabulary: {vocab_size:,} words\n")
    output_file.write("=" * 60 + "\n")

    try:
        similar_words = model.findSynonymsArray(test_word, top_n)
        output_file.write(f"Similar to '{test_word}':\n")
        for i, (word, sim) in enumerate(similar_words, 1):
            output_file.write(f"  {i}. {word} ({sim:.3f})\n")
    except Exception as e:
        output_file.write(f"'{test_word}' not in vocabulary\n")


def save_model_results(model, output_file, results_dir: Path):
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        vectors = model.getVectors()
        vocab_list = [row["word"] for row in vectors.select("word").collect()]

        vocab_file = results_dir / "spark_vocab_simple.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in sorted(vocab_list):
                f.write(f"{word}\n")

        output_file.write(f"Vocabulary saved: {len(vocab_list):,} words\n")
        output_file.write(f"Location: {vocab_file}\n")
    except Exception as e:
        output_file.write(f"Save error: {e}\n")


def main():
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file_path = results_dir / "lab4_spark_word2vec_output.txt"
    data_path = root_dir / "data" / "c4-train.00000-of-01024-30K.json"

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("Lab 4: PySpark Word2Vec Training\n")
        output_file.write("=" * 60 + "\n\n")

        spark = None
        try:
            total_start_time = time.time()

            spark = create_spark_session(output_file)

            if not data_path.exists():
                output_file.write(f"Data file not found: {data_path}\n")
                output_file.write(f"Expected location: {data_path.absolute()}\n")
                return

            processed_df = load_and_preprocess_data(
                spark, data_path, output_file, sample_fraction=1.0
            )
            processed_df.cache()

            model = train_word2vec_model(processed_df, output_file, vector_size=100)

            demonstrate_model_usage(model, output_file, test_word="computer", top_n=5)

            output_file.write("\nTesting additional words:\n")
            output_file.write("-" * 60 + "\n")
            test_words = ["data", "system", "technology", "the"]
            for word in test_words:
                try:
                    similar = model.findSynonymsArray(word, 3)
                    if similar:
                        word_list = ", ".join([w for w, s in similar[:3]])
                        output_file.write(f"Similar to '{word}': {word_list}\n")
                        break
                except Exception:
                    continue

            save_model_results(model, output_file, results_dir)

            total_end_time = time.time()
            total_duration = total_end_time - total_start_time

            output_file.write("\n" + "=" * 60 + "\n")
            output_file.write("SUMMARY\n")
            output_file.write("=" * 60 + "\n")
            output_file.write(
                f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n"
            )
            output_file.write("Training completed successfully!\n")
        except Exception as e:
            output_file.write(f"Error: {e}\n")

        finally:
            if spark:
                spark.stop()
                output_file.write("Spark session stopped\n")

    print(f"Output saved to: {output_file_path}")
    print(f"Absolute path: {output_file_path.absolute()}")


if __name__ == "__main__":
    main()
