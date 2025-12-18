from nlp_restructure.Lab1.Lab1_Tokenization.src.core.interfaces import Tokenizer
from nlp_restructure.Lab1.Lab2_Count_Vectorization.src.core.interfaces import Vectorizer
from typing import List

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}
    
    def fit(self, corpus: List[str]):
        vocab_set = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            vocab_set.update(tokens)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocab_set))}
    
    def transform(self, documents: List[str]) -> List[List[int]]:
        vectors = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                idx = self.vocabulary_.get(token)
                if idx is not None:
                    vector[idx] += 1
            vectors.append(vector)
        return vectors
    
    def fit_transform(self, corpus: List[str]) -> List[List[str]]:
        self.fit(corpus)
        vectors = self.transform(corpus)
        return vectors
