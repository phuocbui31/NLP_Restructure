# Báo cáo Lab 1 & Lab 2

## Source code sử dụng trong báo cáo
- [src/nlp_restructure/Lab1/Lab1_Tokenization/src/core/interfaces.py](../../src/nlp_restructure/Lab1/Lab1_Tokenization/src/core/interfaces.py): Interface Tokenizer
- [src/nlp_restructure/Lab1/Lab2_Count_Vectorization/src/core/interfaces.py](../../src/nlp_restructure/Lab1/Lab2_Count_Vectorization/src/core/interfaces.py): Interface Vectorizer
- [src/nlp_restructure/Lab1/Lab1_Tokenization/src/preprocessing/simple_tokenizer.py](../../src/nlp_restructure/Lab1/Lab1_Tokenization/src/preprocessing/simple_tokenizer.py): SimpleTokenizer
- [src/nlp_restructure/Lab1/Lab1_Tokenization/src/preprocessing/regex_tokenizer.py](../../src/nlp_restructure/Lab1/Lab1_Tokenization/src/preprocessing/regex_tokenizer.py): RegexTokenizer
- [src/nlp_restructure/Lab1/Lab2_Count_Vectorization/src/representations/count_vectorizer.py](../../src/nlp_restructure/Lab1/Lab2_Count_Vectorization/src/representations/count_vectorizer.py): CountVectorizer
- [src/nlp_restructure/Lab1/Lab1_Tokenization/src/core/dataset_loaders.py](../../src/nlp_restructure/Lab1/Lab1_Tokenization/src/core/dataset_loaders.py): Hàm load dữ liệu
- [test/Lab1/main.py](../../test/Lab1/main.py): Test tokenizer
- [test/Lab1/lab2_test.py](../../test/Lab1/lab2_test.py): Test CountVectorizer

## Dataset sử dụng
- **Tên dataset**: Universal Dependencies English-EWT (UD_English-EWT)
- **Mô tả**: Corpus tiếng Anh được gán nhãn ngữ pháp theo chuẩn Universal Dependencies, chứa các câu từ blog, email, review.
- **Cấu trúc dữ liệu**:
  - **Format**: CoNLL-U (tab-separated values)
  - **Số lượng**: ~12,544 câu (train), ~2,001 câu (dev), ~2,077 câu (test)
  - **Các cột chính**:
    - `ID`: Thứ tự token
    - `FORM`: Từ gốc (string)
    - `LEMMA`: Dạng lemma của từ (string)
    - `UPOS`: Universal POS tag (string)
    - `XPOS`: Language-specific POS tag (string)
    - `HEAD`: Head token ID (integer)
    - `DEPREL`: Dependency relation (string)
- **Nguồn**: [Universal Dependencies](https://universaldependencies.org/) - [GitHub UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT)
- **Lưu ý**: Dataset chỉ lưu local tại `data/UD_English-EWT/`.

## 1. Mô tả công việc

### Lab 1
- **Cài đặt interface Tokenizer**: Định nghĩa abstract base class cho các tokenizer.
- **Cài đặt SimpleTokenizer**:
  - Chuyển text về lowercase.
  - Tách token theo whitespace và các dấu câu cơ bản (.,?!).
- **Cài đặt RegexTokenizer**:
  - Sử dụng regex `\w+|[^\w\s]` để tách từ và dấu câu.
  - Xử lý các trường hợp đặc biệt tốt hơn separation cơ bản.

### Lab 2
- **Cài đặt interface Vectorizer**: Định nghĩa phương thức `fit`, `transform`, `fit_transform`.
- **Cài đặt CountVectorizer**:
  - Nhận vào một tokenizer.
  - Xây dựng vocabulary từ corpus.
  - Chuyển văn bản thành vector đếm (Bag of Words).
- **Kiểm thử**:
  - Viết các script test để kiểm tra hoạt động của Tokenizer và Vectorizer trên các câu mẫu và dataset thực tế.

## 2. Cách chạy code và ghi log kết quả

### Cài đặt thư viện uv

*   Đảm bảo cài đặt thư viện `uv`. Nếu chưa cài đặt, có thể sử dụng lệnh: ```curl -LsSf https://astral.sh/uv/install.sh | sh```

*   Khởi tạo môi trường để chạy code:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

*   Cài đặt thư viện cần thiết (bao gồm cả thư viện của các bài thực hành trước):
    ```bash
    uv sync
    ```

Để kiểm thử các chức năng, sử dụng công cụ `uv` với các lệnh sau:

```bash
# Kiểm thử Tokenizer (Lab 1)
uv run -m test.Lab1.main

# Kiểm thử CountVectorizer (Lab 2)
uv run -m test.Lab1.lab2_test
```

## 3. Kết quả chạy code

### Lab 1: Output của các tokenizer

**SimpleTokenizer Results (Sample Texts):**
```
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', 't', 'it', '?']
['let', 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

**RegexTokenizer Results (Sample Texts):**
```
['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

**Dataset Output (UD English EWT - First 5 lines tokenized by RegexTokenizer):**
```
['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani', ',', 'the', 'preacher', 'at', 'the']
['mosque', 'in', 'the', 'town', 'of', 'Qaim', ',', 'near', 'the', 'Syrian', 'border', '.', '[', 'This', 'killing', 'of', 'a', 'respected']
['cleric', 'will', 'be', 'causing', 'us', 'trouble', 'for', 'years', 'to', 'come', '.', ']', 'DPA', ':', 'Iraqi', 'authorities']
['announced', 'that', 'they', 'had', 'busted', 'up', '3', 'terrorist', 'cells', 'operating', 'in', 'Baghdad', '.', 'Two', 'of']
['them', 'were', 'being', 'run', 'by', '2', 'officials', 'of', 'the', 'Ministry', 'of', 'the', 'Interior', '!', 'The', 'MoI', 'in']
```

### Lab 2: Output của CountVectorizer

**Kết quả trên Corpus mẫu:**
```
Learned vocabulary:
{'.': 0, 'AI': 1, 'I': 2, 'NLP': 3, 'a': 4, 'is': 5, 'love': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-term matrix:
[1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
[1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
[1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
```

**Kết quả trên 5 dòng đầu của UD English EWT:**
```
[UD English EWT] Learned vocabulary (first 5 lines):
{'!': 0, ',': 1, '-': 2, '.': 3, '2': 4, '3': 5, ':': 6, 'Abdullah': 7, 'Al': 8, 'American': 9, 'Ani': 10, 'Baghdad': 11, 'DPA': 12, 'Interior': 13, 'Iraqi': 14, 'Ministry': 15, 'MoI': 16, 'Qaim': 17, 'Shaikh': 18, 'Syrian': 19, 'The': 20, 'This': 21, 'Two': 22, 'Zaman': 23, '[': 24, ']': 25, 'a': 26, 'al': 27, 'announced': 28, 'at': 29, 'authorities': 30, 'be': 31, 'being': 32, 'border': 33, 'busted': 34, 'by': 35, 'causing': 36, 'cells': 37, 'cleric': 38, 'come': 39, 'for': 40, 'forces': 41, 'had': 42, 'in': 43, 'killed': 44, 'killing': 45, 'mosque': 46, 'near': 47, 'of': 48, 'officials': 49, 'operating': 50, 'preacher': 51, 'respected': 52, 'run': 53, 'terrorist': 54, 'that': 55, 'the': 56, 'them': 57, 'they': 58, 'to': 59, 'town': 60, 'trouble': 61, 'up': 62, 'us': 63, 'were': 64, 'will': 65, 'years': 66}

[UD English EWT] Document-term matrix (first 5 lines):
[0, 1, 2, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

## 4. Giải thích kết quả

- **SimpleTokenizer**: Chỉ tách các dấu câu cơ bản. Ví dụ: `Hello, world!` -> `['Hello', ',', 'world', '!']`. Tuy nhiên, các contract của tiếng Anh như `isn't` không được xử lý tốt (bị tách thành `isn`, `t`).
- **RegexTokenizer**: Dùng biểu thức chính quy `\w+|[^\w\s]` để tách mọi token là từ hoặc ký tự đặc biệt. Kết quả cho thấy nó giữ nguyên được các từ và tách riêng biệt các dấu câu liền kề. Ví dụ: `isn't` được xử lý thành `isn`, `'`, `t` (tốt hơn một chút nhưng vẫn chưa hoàn hảo cho phân tích ngữ nghĩa nếu không có luật normalization).
- **CountVectorizer**:
  - Với corpus mẫu nhỏ, vocabulary có 10 từ. Ma trận thưa (sparse matrix) nhưng ở đây hiển thị dạng dense list.
  - Với dataset UD (chỉ lấy 5 dòng đầu), vocabulary lớn hơn (67 từ) và ma trận có nhiều số 0 (thể hiện tính thưa thớt của dữ liệu văn bản).

## 5. Khó khăn và cách giải quyết

- **Khó khăn:** Xử lý tiếng Anh với các ký tự đặc biệt, các dấu câu liên tiếp
- **Cách giải quyết:** Cần chỉnh sửa Regex, logic tokenizer

## 6. Nguồn tham khảo
- [Python re module documentation](https://docs.python.org/3/library/re.html)
- [Scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- Tài liệu giảng dạy trên lớp

## 7. Model tạo sẵn, prompt sử dụng
- **Model**: Không sử dụng model tạo sẵn bên ngoài.
- **Dataset**: UD English EWT.
