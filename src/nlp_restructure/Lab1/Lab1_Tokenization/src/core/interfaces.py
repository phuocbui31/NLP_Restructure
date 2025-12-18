from typing import List
from abc import ABC, abstractmethod

class Tokenizer(ABC):    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses should implement this method.")
