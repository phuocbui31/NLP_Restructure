from nlp_restructure.Lab1.Lab1_Tokenization.src.preprocessing.regex_tokenizer import RegexTokenizer
from nlp_restructure.Lab1.Lab2_Count_Vectorization.src.representations.count_vectorizer import CountVectorizer
from nlp_restructure.Lab1.Lab1_Tokenization.src.core.dataset_loaders import load_raw_text_data

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

X = vectorizer.fit_transform(corpus)

print("Learned vocabulary:")
print(vectorizer.vocabulary_)
print("\nDocument-term matrix:")
for row in X:
    print(row)

ud_path = "data/UD_English-EWT/en_ewt-ud-train.txt"
ud_text = load_raw_text_data(ud_path)
ud_corpus = [line for line in ud_text.split('\n') if line.strip()]

ud_vectorizer = CountVectorizer(tokenizer)
ud_X = ud_vectorizer.fit_transform(ud_corpus[:5])

print("\n[UD English EWT] Learned vocabulary (first 5 lines):")
print(ud_vectorizer.vocabulary_)
print("\n[UD English EWT] Document-term matrix (first 5 lines):")
for row in ud_X:
    print(row)
