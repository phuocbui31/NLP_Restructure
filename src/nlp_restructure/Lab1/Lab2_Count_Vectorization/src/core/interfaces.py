from typing import List
from abc import ABC, abstractmethod

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        raise NotImplementedError("Subclasses should implement this method")
