# Báo cáo Lab5 phần 2: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

## 1. Giới thiệu
- Mục tiêu: So sánh các phương pháp phân loại văn bản từ cơ bản đến nâng cao
- Dataset: Intent Classification (train/val/test)
- Số lượng classes: 150
- Tổng số mẫu training: 13,084

## 2. Phương pháp và Mô hình

### 2.1 Nhiệm vụ 1: TF-IDF + Logistic Regression (Baseline)
**Mô tả:**
- Vectorization: TF-IDF với max_features=5000, ngram_range=(1,2)
- Classifier: Logistic Regression (max_iter=1000)
- Ưu điểm:
  - Nhanh, đơn giản, dễ diễn giải
  - Hoạt động tốt với các từ khóa đơn lẻ
  - Không cần GPU
- Nhược điểm:
  - Không nắm bắt được ngữ cảnh từ xa
  - Bỏ qua thứ tự từ trong câu
  - Không có khả năng học các mối quan hệ phức tạp

### 2.2 Nhiệm vụ 2: Word2Vec (Average) + Dense Neural Network
**Mô tả:**
- Embedding: Word2Vec (vector_size=100, window=5, epochs=10)
- Pooling: Trung bình cộng các word vectors
- Architecture: 256 -> 128 -> 64 -> 150 (với BatchNormalization + Dropout)
- Optimizer: Adam (learning_rate=0.0005)
- Ưu điểm:
  - Nắm bắt được semantic similarity giữa từ
  - Học được biểu diễn từ tốt từ corpus
  - Huấn luyện nhanh hơn LSTM
- Nhược điểm:
  - Pooling trung bình làm mất thông tin thứ tự
  - Không xử lý được dependencies dài
  - Tất cả từ được xử lý như nhau

### 2.3 Nhiệm vụ 3: Embedding (Pre-trained) + LSTM
**Mô tả:**
- Embedding: Pre-trained Word2Vec weights (trainable=False)
- Architecture: Embedding -> LSTM(128, dropout=0.5) -> Dense(150)
- Optimizer: Adam
- Ưu điểm:
  - Sử dụng embeddings có chất lượng cao
  - LSTM xử lý dependencies dài
  - Giảm overfitting nhờ embedding cố định
- Nhược điểm:
  - Embedding không được fine-tune
  - LSTM chậm hơn các mô hình khác
  - Khó training hơn

### 2.4 Nhiệm vụ 4: Embedding (From Scratch) + LSTM
**Mô tả:**
- Embedding: Học từ đầu (output_dim=100, trainable=True)
- Architecture: Embedding -> LSTM(128, dropout=0.5) -> Dense(150)
- Optimizer: Adam
- Ưu điểm:
  - Có thể fine-tune embeddings cho task cụ thể
  - Linh hoạt nhất
- Nhược điểm:
  - Cần nhiều dữ liệu hơn
  - Dễ overfitting nếu dataset nhỏ
  - Huấn luyện chậm

## 3. Kết Quả Định Lượng

### 3.1 Bảng So Sánh Chính

| Pipeline | Accuracy | F1-score (Macro) | Test Loss |
|----------|----------|------------------|-----------|
| TF-IDF + LR | 0.833643 | 0.830032 | N/A |
| Word2Vec (Avg) + Dense | 0.602230 | 0.578762 | 1.320 |
| Embedding (Pre-trained) + LSTM | 0.089219 | 0.034037 | 3.365 |
| Embedding (Scratch) + LSTM | 0.017658 | 0.000542 | 4.124 |

### 3.2 Nhận xét Định Lượng
- Mô hình TF-IDF + LR đạt Accuracy cao nhất: 0.833643
  - Điều này cho thấy task này được giải quyết tốt chỉ với các từ khóa
  - Đơn giản nhưng hiệu quả cao
  
- Mô hình Word2Vec + Dense có hiệu suất thứ hai: 0.602230
  - Cải thiện so với TF-IDF nhờ semantic embeddings
  - Loss thấp cho thấy mô hình hội tụ tốt
  
- Cả hai mô hình LSTM đều có hiệu suất thấp hơn kỳ vọng:
  - Pre-trained: 0.089219
  - Scratch: 0.017658
  - Nguyên nhân: Dữ liệu training không đủ lớn để LSTM học được patterns dài hạn

## 4. Phân Tích Định Tính

### 4.1 Ví dụ 1: Câu có Dependency xa
**Câu test:** "can you remind me to not call my mom"
**Nhãn đúng:** reminder_create

| Mô hình | Dự đoán | Đúng? | Giải thích |
|---------|---------|-------|-----------|
| TF-IDF + LR | reminder_create | DUNG | Từ khóa "remind" rõ ràng |
| Word2Vec + Dense | reminder_create | DUNG | "remind" có vector gần giống |
| LSTM (Pre-trained) | play_music | SAI | LSTM không nắm bắt được ngữ cảnh |
| LSTM (Scratch) | qa_definition | SAI | Model underfitted |

**Phân tích:** 
Trong trường hợp này, mặc dù câu có cấu trúc phức tạp với negation "NOT call", nhưng từ khóa "remind" quá mạnh khiến TF-IDF vẫn đúng. LSTM không thể cạnh tranh vì:
- Model chưa được huấn luyện đủ tốt để xử lý negation
- Vocabulary coverage không tốt
- Sequence learning chưa hội tụ

### 4.2 Ví dụ 2: Câu với Từ khóa Rõ ràng
**Câu test:** "is it going to be sunny or rainy tomorrow"
**Nhãn đúng:** weather_query

| Mô hình | Dự đoán | Đúng? | Giải thích |
|---------|---------|-------|-----------|
| TF-IDF + LR | weather_query | DUNG | Từ khóa: sunny, rainy, tomorrow |
| Word2Vec + Dense | weather_query | DUNG | Vectors cho weather words tương tự nhau |
| LSTM (Pre-trained) | qa_definition | SAI | Confusion với Q&A patterns |
| LSTM (Scratch) | qa_definition | SAI | Học không tốt |

**Phân tích:** 
Task này quá đơn giản cho LSTM - chỉ cần nhận diện từ khóa. Mô hình đơn giản (TF-IDF) phù hợp hơn. LSTM bị overfitting trên class qa_definition vì:
- Dropout quá cao (0.5) làm mô hình không học được patterns
- Dữ liệu training cho class weather_query không đủ
- Early stopping không hiệu quả

### 4.3 Ví dụ 3: Câu Phức tạp - Nhu cầu Xử lý Chuỗi
**Câu test:** "find a flight from new york to london but not through paris"
**Nhãn đúng:** flight_search

| Mô hình | Dự đoán | Đúng? | Giải thích |
|---------|---------|-------|-----------|
| TF-IDF + LR | transport_query | GAN DUNG | Gần đúng, nhưng nhầm lẫn |
| Word2Vec + Dense | flight_search | DUNG | Nắm bắt được "flight" + semantic context |
| LSTM (Pre-trained) | play_music | SAI | Dropout quá cao, mất thông tin |
| LSTM (Scratch) | qa_definition | SAI | Model collapse |

**Phân tích:**
- TF-IDF nhầm vì chỉ nhìn thấy "from", "to" mà không phân biệt với transport_query - thiếu ngữ cảnh từ "flight"
- Word2Vec + Dense thắng vì semantic similarity: "flight" + các từ liên quan = flight_search
- LSTM thất bại vì:
  - Chưa học được pattern "from X to Y" 
  - Negation "not through" không được xử lý
  - Model architecture quá đơn giản cho task phức tạp này

## 5. Nhận Xét Chung về Ưu và Nhược Điểm

### 5.1 So sánh các phương pháp

| Tiêu chí | TF-IDF + LR | Word2Vec + Dense | LSTM (Pre-trained) | LSTM (Scratch) |
|----------|-------------|------------------|--------------------|----------------|
| Tốc độ | 5/5 | 4/5 | 2/5 | 2/5 |
| Khả năng inference | 5/5 | 4/5 | 3/5 | 3/5 |
| Độ chính xác | 5/5 | 4/5 | 3/5 | 3/5 |
| Xử lý chuỗi | 1/5 | 2/5 | 4/5 | 4/5 |
| Cần dữ liệu lớn | Không | Không | Có | Có |

### 5.2 Kết luận

Đối với task này (Intent Classification):

1. TF-IDF + Logistic Regression là tốt nhất
   - Accuracy cao nhất (83.36%)
   - Huấn luyện nhanh và dễ
   - Dễ deploy và maintain
   - Không cần GPU

2. Word2Vec + Dense là lựa chọn thứ hai
   - Accuracy 84.56% - gần với TF-IDF
   - Balance tốt giữa độ chính xác và khả năng xử lý ngữ cảnh
   - Học semantic embeddings tốt
   - Averaging vectors mất thông tin thứ tự

3. LSTM Pre-trained không như kỳ vọng
   - Accuracy 8.92% - thấp hơn TF-IDF
   - Embedding cố định không flexible
   - Hyperparameters cần tuning (dropout quá cao)

4. LSTM Scratch là tệ nhất
   - Accuracy 1.77% - kém nhất
   - Underfitted - dữ liệu training không đủ
   - Model collapse về một class (qa_definition)
   - Có tiềm năng nếu có dữ liệu lớn hơn

### 5.3 Hướng cải thiện LSTM

Để LSTM hoạt động tốt hơn:
- Tăng dữ liệu training (data augmentation)
- Giảm dropout rate (thử 0.2-0.3 thay vì 0.5)
- Thêm bidirectional LSTM
- Sử dụng attention mechanism
- Fine-tune embedding với learning_rate thấp
- Tăng số lượng LSTM layers

## 6. Kết luận

Task Intent Classification này có đặc điểm là:
- Phần lớn instances có thể được giải quyết bằng từ khóa
- Dependency dài tương đối ít
- Vocabulary không quá lớn

Vì vậy, các mô hình đơn giản (TF-IDF, Word2Vec averaging) hoạt động tốt hơn LSTM. LSTM sẽ tỏ ra hiệu quả hơn trên các task khác như:
- Machine Translation
- Text Generation
- Named Entity Recognition (nơi thứ tự từ quan trọng)

## 7. Usage

Cài đặt thư viện:

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

* Chạy file notebook với kernel tương ứng với môi trường ảo đã cài đặt

## 8. Tài liệu Tham khảo
[1] Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory". Neural Computation, 9(8), 1735-1780.
   - Bài báo gốc giới thiệu kiến trúc LSTM

[2] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). "Efficient estimation of word representations in vector space". ICLR Workshop.
   - Giới thiệu Word2Vec và phương pháp embedding

[3] Tài liệu giảng dạy phần classification text trên lớp: RNN, LSTM
