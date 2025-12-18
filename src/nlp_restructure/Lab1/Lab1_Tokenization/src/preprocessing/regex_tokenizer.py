import re
from typing import List

from nlp_restructure.Lab1.Lab1_Tokenization.src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
