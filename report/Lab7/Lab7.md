# Báo cáo Lab 7 — Dependency Parsing

## 1. Mục tiêu
- Sử dụng thư viện spaCy để thực hiện phân tích cú pháp phụ thuộc cho một câu.
- Trực quan hóa cây phụ thuộc để hiểu rõ cấu trúc câu.
- Truy cập và duyệt (traverse) cây phụ thuộc theo chương trình.
- Trích xuất thông tin có ý nghĩa từ các mối quan hệ phụ thuộc (ví dụ: tìm chủ ngữ, tân ngữ, bổ ngữ).

---

## Link to file code:
[https://github.com/phuocbui31/NLP/blob/main/Lab7/Lab7_Dependency_Parsing.pdf](https://github.com/phuocbui31/NLP/blob/main/Lab7/Lab7_Dependency_Parsing.pdf)

## 2. Phần 2: Phân tích câu và Trực quan hóa

### Câu hỏi & Trả lời

1. **Từ nào là gốc (ROOT) của câu?**  
   - Trong câu "The quick brown fox jumps over the lazy dog.", từ "jumps" là gốc (ROOT) của câu. Đây là động từ chính, có quan hệ dependency là "ROOT".

2. **jumps có những từ phụ thuộc (dependent) nào? Các quan hệ đó là gì?**  
   - "jumps" có các từ phụ thuộc:
     - "fox" với quan hệ `nsubj` (nominal subject - chủ ngữ)
     - "over" với quan hệ `prep` (prepositional modifier - giới từ)
     - "." với quan hệ `punct` (punctuation - dấu câu)

3. **fox là head của những từ nào?**  
   - "fox" là head của:
     - "The" với quan hệ `det` (determiner - từ xác định)
     - "quick" với quan hệ `amod` (adjectival modifier - tính từ bổ nghĩa)
     - "brown" với quan hệ `amod` (adjectival modifier - tính từ bổ nghĩa)


## 3. Phần 3: Truy cập các thành phần trong cây phụ thuộc

### Giải thích các thuộc tính

- `token.text`: Văn bản của token.
- `token.dep_`: Nhãn quan hệ phụ thuộc của token này với head của nó.
- `token.head.text`: Văn bản của token head.
- `token.head.pos_`: Part-of-Speech tag của token head.
- `token.children`: Một iterator chứa các token con (dependent) của token hiện tại.

### Kết quả phân tích câu "Apple is looking at buying U.K. startup for $1 billion"

Kết quả cho thấy:
- "looking" là ROOT của câu
- "Apple" là chủ ngữ (nsubj) của "looking"
- "at" là giới từ (prep) phụ thuộc vào "looking"
- "buying" là complement của giới từ (pcomp) phụ thuộc vào "at"
- "startup" là tân ngữ trực tiếp (dobj) của "buying"
- "U.K." là compound phụ thuộc vào "startup"
- "for" là giới từ phụ thuộc vào "startup"
- "$1 billion" là cụm từ phụ thuộc vào "for"

## 4. Phần 4: Duyệt cây phụ thuộc để trích xuất thông tin

### 4.1. Bài toán: Tìm chủ ngữ và tân ngữ của một động từ

**Kết quả:**  
Với câu "The cat chased the mouse and the dog watched them.", hàm tìm được:
- Found Triplet: (cat, chased, mouse)

**Giải thích:**  
- "chased" là động từ có chủ ngữ "cat" (nsubj) và tân ngữ "mouse" (dobj)
- "watched" cũng là động từ nhưng chỉ có chủ ngữ "dog", không có tân ngữ trực tiếp (dobj) nên không được in ra

### 4.2. Bài toán: Tìm các tính từ bổ nghĩa cho một danh từ

**Kết quả:**  
Với câu "The big, fluffy white cat is sleeping on the warm mat.", hàm tìm được:
- Danh từ 'cat' được bổ nghĩa bởi các tính từ: ['big', 'fluffy', 'white']
- Danh từ 'mat' được bổ nghĩa bởi các tính từ: ['warm']

**Giải thích:**  
- Các tính từ có quan hệ `amod` (adjectival modifier) với danh từ
- Một danh từ có thể có nhiều tính từ bổ nghĩa

---

## 5. Phần 5: Bài tập tự luyện

### Bài 1: Tìm động từ chính của câu

**Hàm `find_main_verb(doc)`:**
- Tìm token có quan hệ `ROOT` và POS tag là `VERB`
- Trả về Token là động từ chính, hoặc `None` nếu không tìm thấy

**Kết quả:**  
Với câu "The student studied hard and passed the exam.", hàm tìm được động từ chính là "studied" (động từ đầu tiên có quan hệ ROOT).

### Bài 2: Trích xuất các cụm danh từ (Noun Chunks)

**Hàm `extract_noun_chunks(doc)`:**
- Tìm tất cả các danh từ trong câu
- Với mỗi danh từ, thu thập các từ bổ nghĩa (det, amod, compound)
- Sắp xếp các từ theo thứ tự trong câu
- Trả về danh sách các cụm danh từ

**Kết quả:**  
Với câu "The big, fluffy white cat is sleeping on the warm mat.", hàm tìm được:
- "The big fluffy white cat"
- "the warm mat"

**So sánh với spaCy:**  
Kết quả từ hàm tự viết có thể khác với `doc.noun_chunks` của spaCy vì spaCy sử dụng thuật toán phức tạp hơn để xác định ranh giới cụm danh từ.

### Bài 3: Tìm đường đi ngắn nhất trong cây

**Hàm `get_path_to_root(token)`:**
- Bắt đầu từ token đã cho
- Đi ngược lên head cho đến khi gặp ROOT
- Trả về danh sách các token trên đường đi

**Kết quả:**  
Với câu "The quick brown fox jumps over the lazy dog." và token "dog":
- Đường đi từ 'dog' lên ROOT:
  1. dog (pobj)
  2. over (prep)
  3. jumps (ROOT)

**Giải thích:**  
- "dog" phụ thuộc vào "over" (pobj)
- "over" phụ thuộc vào "jumps" (prep)
- "jumps" là ROOT của câu

---

## 6. Kết luận

- Dependency parsing cho phép hiểu cấu trúc ngữ pháp của câu dưới dạng các mối quan hệ head-dependent.
- spaCy cung cấp các công cụ mạnh mẽ để phân tích và trực quan hóa dependency trees.
- Việc truy cập và duyệt cây phụ thuộc theo chương trình cho phép trích xuất thông tin ngữ nghĩa quan trọng như chủ ngữ, tân ngữ, và các từ bổ nghĩa.
- Các bài tập tự luyện giúp nắm vững cách sử dụng các thuộc tính của token để giải quyết các bài toán NLP thực tế.

## 7. Tài liệu tham khảo

- spaCy Documentation: https://spacy.io/
- spaCy Dependency Parser: https://spacy.io/usage/linguistic-features#dependency-parse
- Universal Dependencies: https://universaldependencies.org/
- Honnibal, M. & Montani, I., *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing*, 2017.
- Tài liệu giảng dạy trên lớp
