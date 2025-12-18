# Báo cáo Lab 6 part 1 — Intro to Transformers

## 1. Mục tiêu
- Thực hành Masked Language Modeling (MLM), Next Token Prediction và xây dựng sentence representation bằng Transformers.
- Hiểu sự khác biệt giữa encoder-only (BERT) và decoder-only (GPT) architectures.
- Thực hành kỹ thuật pooling có xét padding (attention_mask).

---

## Link to file code:
[https://github.com/phuocbui31/NLP/blob/main/Lab6/Lab6_Intro_to_Transformers.pdf](https://github.com/phuocbui31/NLP/blob/main/Lab6/Lab6_Intro_to_Transformers.pdf)

## 2. Bài 1 — Masked Language Modeling

### Câu hỏi & Trả lời
1. Mô hình đã dự đoán đúng từ "capital" không?  
   - Có. Pipeline `fill-mask` (mặc định là biến thể BERT) dự đoán được từ "capital" cho câu "Hanoi is the <mask> of Vietnam.".

2. Tại sao các mô hình encoder-only như BERT phù hợp cho tác vụ này?  
   - BERT được pretrain bằng MLM và học biểu diễn ngữ cảnh hai chiều (bidirectional). Khi một token bị mask, mô hình sử dụng ngữ cảnh cả bên trái và bên phải để suy đoán token đó, do đó phù hợp cho việc khôi phục masked tokens.

---

## 3. Bài 2 — Next Token Prediction (ghi chú ngắn)
- Decoder-only models (ví dụ GPT) học theo cơ chế autoregressive (causal). Chúng chỉ dùng ngữ cảnh các token trước để sinh token tiếp theo, phù hợp cho sinh văn bản và dự đoán token kế tiếp.

---

## 4. Bài 3 — Sentence Representation

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
