import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import sys

import gensim.downloader as api
from loguru import logger

root_dir = Path(__file__).parents[3]
sys.path.insert(0, str(root_dir))


class WordEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}...")
        self.model = api.load(model_name)
        logger.success(f"Model loaded successfully! Vocabulary size: {len(self.model.key_to_index):,}")
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            logger.warning(f"Warning: '{word}' is not in vocabulary (OOV word)")
            return None
    
    def get_similarity(self, w1: str, w2: str) -> Optional[float]:
        if w1 not in self.model.key_to_index:
            logger.warning(f"Warning: '{w1}' is not in vocabulary (OOV word)")
            return None
        if w2 not in self.model.key_to_index:
            logger.warning(f"Warning: '{w2}' is not in vocabulary (OOV word)")
            return None
        
        return self.model.similarity(w1, w2)
    
    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if word not in self.model.key_to_index:
            logger.warning(f"Warning: '{word}' is not in vocabulary (OOV word)")
            return []
        
        return self.model.most_similar(word, topn=top_n)
    
    def get_vector_dimension(self) -> int:
        return self.model.vector_size

    def embed_document(self, document: str) -> np.ndarray:
        tokens: List[str]
        try:
            from Lab1.Lab1_Tokenization.src.preprocessing.regex_tokenizer import RegexTokenizer
            tokenizer = RegexTokenizer()
            tokens = tokenizer.tokenize(document)
        except Exception:
            import re
            tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", document.lower())]

        vecs = []
        for tok in tokens:
            v = self.get_vector(tok)
            if v is not None:
                vecs.append(np.asarray(v, dtype=float))

        dim = self.get_vector_dimension()
        if len(vecs) == 0:
            return np.zeros(dim, dtype=float)

        stacked = np.vstack(vecs)
        doc_vec = np.mean(stacked, axis=0)
        return doc_vec
