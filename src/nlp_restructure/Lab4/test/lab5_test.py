import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from Lab1.Lab1_Tokenization.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab1.Lab2_Count_Vectorization.src.representations.count_vectorizer import CountVectorizer
from Lab4.src.models.text_classifier import TextClassifier

# Define the texts and labels dataset
texts = [
	"This movie is fantastic and I love it!",
	"I hate this film, it's terrible.",
	"The acting was superb, a truly great experience.",
	"What a waste of time, absolutely boring.",
	"Highly recommend this, a masterpiece.",
	"Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Instantiate your RegexTokenizer and TfidfVectorizer
tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer=tokenizer)

#Instantiate your TextClassifier with the vectorizer
classifier = TextClassifier(vectorizer)

# Train the classifier using the training data
start_time = time.time()
classifier.fit(X_train, y_train)
train_time = time.time() - start_time

# Make predictions on the test data
start_pred_time = time.time()
y_pred = classifier.predict(X_test)
pred_time = time.time() - start_pred_time

# Evaluate
metrics = classifier.evaluate(y_test, y_pred)

outputs = []
outputs.append(f"Training time: {train_time:.3f} seconds")
outputs.append(f"Prediction time: {pred_time:.3f} seconds")
outputs.append("Evaluation metrics:")
for k, v in metrics.items():
	outputs.append(f"{k}: {v:.3f}")

# Save to the file
result_dir = Path(__file__).parents[1] / 'results'
result_dir.mkdir(parents=True, exist_ok=True)
output_path = result_dir / "lab5_test_results.txt"

with open(output_path, 'w', encoding='utf-8') as f:
	f.write('\n'.join(outputs))

# Print the metrics
for line in outputs:
	print(line)
