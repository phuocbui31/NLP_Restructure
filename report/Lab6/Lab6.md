# Báo cáo Lab 6 part 1 — Intro to Transformers

## 1. Mục tiêu
- Thực hành Masked Language Modeling (MLM), Next Token Prediction và xây dựng sentence representation bằng Transformers.
- Hiểu sự khác biệt giữa encoder-only (BERT) và decoder-only (GPT) architectures.
- Thực hành kỹ thuật pooling có xét padding (attention_mask).

---

### Files trong project
- [Notebook thực hành (Lab6_Intro_to_Transformers.ipynb)](../../notebook/Lab6/Lab6_Intro_to_Transformers.ipynb)
- [PDF hướng dẫn (Lab6_Intro_to_Transformers.pdf)](../../notebook/Lab6/Lab6_Intro_to_Transformers.pdf)

## 2. Bài 1 — Masked Language Modeling

### Dữ liệu & Code Implementation
- **Dữ liệu**: Câu đầu vào là "Hanoi is the <mask> of Vietnam."
- **Code**:
```python
from transformers import pipeline

# 1. Tải pipeline "fill-mask"
# Pipeline này sẽ tự động tải một mô hình mặc định phù hợp (thường là một biến thể của BERT)
mask_filler = pipeline("fill-mask")

# 2. Câu đầu vào với token <mask>
input_sentence = "Hanoi is the <mask> of Vietnam."

# 3. Thực hiện dự đoán
# top_k=5 yêu cầu mô hình trả về 5 dự đoán hàng đầu
predictions = mask_filler(input_sentence, top_k=5)

# 4. In kết quả
print(f"Câu gốc: {input_sentence}")

for pred in predictions:
  print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
  print(f" -> Câu hoàn chỉnh: {pred['sequence']}")
```

### Câu hỏi & Trả lời
1. Mô hình đã dự đoán đúng từ "capital" không?  
   - Có. Pipeline `fill-mask` (mặc định là biến thể BERT) dự đoán được từ "capital" cho câu "Hanoi is the <mask> of Vietnam.".

2. Tại sao các mô hình encoder-only như BERT phù hợp cho tác vụ này?  
   - BERT được pretrain bằng MLM và học biểu diễn ngữ cảnh hai chiều (bidirectional). Khi một token bị mask, mô hình sử dụng ngữ cảnh cả bên trái và bên phải để suy đoán token đó, do đó phù hợp cho việc khôi phục masked tokens.

---

## 3. Bài 2 — Next Token Prediction (ghi chú ngắn)

### Dữ liệu & Code Implementation
- **Dữ liệu**: Prompt "The best thing about learning NLP is"
- **Code**:
```python
from transformers import pipeline

# 1. Tải pipeline "text-generation"
# Pipeline này sẽ tự động tải một mô hình phù hợp (thường là GPT-2)
generator = pipeline("text-generation")

# 2. Đoạn văn bản mồi
prompt = "The best thing about learning NLP is"

# 3. Sinh văn bản
# max_length: tổng độ dài của câu mồi và phần được sinh ra
# num_return_sequences: số lượng chuỗi kết quả muốn nhận
generated_texts = generator(prompt, max_length=50, num_return_sequences=1)

# 4. In kết quả
print(f"Câu mồi: '{prompt}'")
for text in generated_texts:
  print("Văn bản được sinh ra:")
  print(text['generated_text'])
```

- Decoder-only models (ví dụ GPT) học theo cơ chế autoregressive (causal). Chúng chỉ dùng ngữ cảnh các token trước để sinh token tiếp theo, phù hợp cho sinh văn bản và dự đoán token kế tiếp.

---

## 4. Bài 3 — Sentence Representation

### Dữ liệu & Code Implementation
- **Dữ liệu**: Câu đầu vào "This is a sample sentence."
- **Code**:
```python
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Chọn một mô hình BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Câu đầu vào
sentences = ["This is a sample sentence."]

# 3. Tokenize câu
# padding=True: đệm các câu ngắn hơn để có cùng độ dài
# truncation=True: cắt các câu dài hơn
# return_tensors='pt': trả về kết quả dưới dạng PyTorch tensors
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 4. Đưa qua mô hình để lấy hidden states
# torch.no_grad() để không tính toán gradient, tiết kiệm bộ nhớ
with torch.no_grad():
  outputs = model(**inputs)

# outputs.last_hidden_state chứa vector đầu ra của tất cả các token
last_hidden_state = outputs.last_hidden_state
# shape: (batch_size, sequence_length, hidden_size)

# 5. Thực hiện Mean Pooling
# Để tính trung bình chính xác, chúng ta cần bỏ qua các token đệm (padding tokens)
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
sentence_embedding = sum_embeddings / sum_mask

# 6. In kết quả
print("Vector biểu diễn của câu:")
print(sentence_embedding)
print("\nKích thước của vector:", sentence_embedding.shape)
```

### Câu hỏi & Trả lời
1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?  
   - Kích thước thu được trong ví dụ là 768. Đây tương ứng với `hidden_size` (số chiều của hidden states) trong cấu hình BERT (ví dụ `bert-base` có `hidden_size=768`).

2. Tại sao cần sử dụng `attention_mask` khi thực hiện Mean Pooling?  
   - `attention_mask` phân biệt token thực (1) và token padding (0). Khi tính mean pooling phải loại trừ embeddings tương ứng padding để không làm loãng vector câu; cụ thể nhân embedding với mask, tổng hợp và chia cho tổng mask (số token thực).

---

## 5. Kết luận ngắn
- BERT (encoder-only) là lựa chọn tự nhiên cho MLM vì khả năng học ngữ cảnh hai chiều.  
- GPT (decoder-only) là lựa chọn cho sinh ngôn ngữ autoregressive.  
- Khi xây sentence embedding từ token hidden states, luôn loại trừ padding bằng `attention_mask` để đảm bảo biểu diễn chính xác; kích thước embedding phụ thuộc trực tiếp vào `hidden_size` của model.

## 6. Tài liệu tham khảo
- Devlin J., Chang M.-W., Lee K., Toutanova K., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2019.  
- Radford A. et al., *GPT: Improving Language Understanding by Generative Pre-Training*.  
- Hugging Face Transformers — https://huggingface.co/docs/transformers  
- "Attention is All You Need" — Vaswani et al., 2017.
- Tài liệu giáng dạy trên lớp
