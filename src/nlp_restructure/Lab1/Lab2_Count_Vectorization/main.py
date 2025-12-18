from ..Lab1_Tokenization.src.core.interfaces import Vectorizer
from Lab2_Count_Vectorization.src.representations.count_vectorizer import CountVectorizer

def main():
    vectorizer = CountVectorizer()
    
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
        ]
    vectorizer.fit(corpus)
    print("This is Lab 2 main function.")

if __name__ == "__main__":
    main()
