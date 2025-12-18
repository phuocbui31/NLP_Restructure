from pathlib import Path

from nlp_restructure.Lab1.Lab1_Tokenization.src.preprocessing.simple_tokenizer import SimpleTokenizer
from nlp_restructure.Lab1.Lab1_Tokenization.src.preprocessing.regex_tokenizer import RegexTokenizer

def main():
    datatset_path = "data/UD_English-EWT/en_ewt-ud-train.txt"
    
    if not Path(datatset_path).exists():
        raise FileNotFoundError(f"Dataset path {datatset_path} does not exist.")
    
    tokenizer = RegexTokenizer()
    
    try:
        with open(datatset_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                if line_count >= 5:
                    break
                
                if not line or line.startswith("#"):
                    continue
                
                tokens = tokenizer.tokenize(line.strip())
                print(tokens)
                line_count += 1
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the dataset: {e}")

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    text1 = "Hello, world! This is a test."
    text2 = "NLP is fascinating... isn't it?"
    text3 = "Let's see how it handles 123 numbers and punctuation!"

    for text in [text1, text2, text3]:
        tokens = simple_tokenizer.tokenize(text=text)
        print(tokens)

    for text in [text1, text2, text3]:
        tokens = regex_tokenizer.tokenize(text=text)
        print(tokens)
        
    main()
