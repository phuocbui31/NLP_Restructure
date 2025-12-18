def load_raw_text_data(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
