
# Báo cáo Lab 5 Part 2: RNNs cho Phân loại Văn bản

## 1. Mô tả Dữ liệu (Dataset Description)

Bộ dữ liệu được sử dụng cho bài thực hành này bao gồm các câu văn bản được phân loại vào 64 nhóm ý định (intents) khác nhau. Dữ liệu được chia thành 3 tập:
-   **Train**: Tập huấn luyện.
-   **Validation**: Tập kiểm định.
-   **Test**: Tập kiểm tra.

Dữ liệu đầu vào là văn bản thô, cần được tiền xử lý (tokenization, padding) trước khi đưa vào các mô hình học sâu.

## 2. Cấu trúc Project

Mã nguồn thực hiện các thí nghiệm nằm trong notebook:
-   [Lab05_RNNs_Text_Classification_.ipynb](../../notebook/Lab5/part2/Lab05_RNNs_Text_Classification_.ipynb)

## 3. Kết quả Thực nghiệm

Dưới đây là tóm tắt kết quả của 4 mô hình được thử nghiệm:

### Task 1: TF-IDF + Logistic Regression (Warm-up)
Mô hình cơ sở sử dụng đặc trưng truyền thống TF-IDF kết hợp với bộ phân loại Logistic Regression.

**Code:**
```python
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```

**Kết quả:**
-   **Test F1-score**: 0.65
-   *Nhận xét*: Đây là kết quả tốt nhất trong số tất cả các mô hình được thử nghiệm, cho thấy mô hình truyền thống vẫn rất hiệu quả với bộ dữ liệu này.

### Task 2: Word2Vec + Dense Neural Network (Warm-up)
Sử dụng Embedding từ Word2Vec (trung bình cộng các vector từ) đưa vào mạng Dense đơn giản.

**Code:**
```python
word2vec_dense_model = Sequential([
    Input(shape=(300,)),  # Word2Vec vector size
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

**Kết quả:**
-   **Test Accuracy**: 0.018
-   **Test F1-score**: 0.001
-   *Nhận xét*: Kết quả rất thấp, mô hình không học được các đặc trưng hiệu quả từ trung bình vector Word2Vec.

### Task 3: Embedding (Pre-trained) + LSTM
Mô hình sử dụng lớp Embedding khởi tạo từ trọng số Pre-trained, kết hợp với mạng LSTM để nắm bắt thông tin chuỗi.

**Code:**
```python
lstm_pretrained_model = Sequential([
    Embedding(
        input_dim=vocab_size + 1,
        output_dim=300,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False
    ),
    LSTM(128, dropout=0.5, recurrent_dropout=0),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

**Kết quả:**
-   **Test Accuracy**: 0.089
-   **Test F1-score**: 0.034
-   *Nhận xét*: Hiệu suất có cải thiện so với Task 2 nhưng vẫn rất thấp. Việc sử dụng embedding pre-trained chưa phát huy hiệu quả mong đợi có thể do sự khác biệt miền dữ liệu hoặc kích thước mô hình chưa phù hợp.

### Task 4: Embedding (From Scratch) + LSTM
Mô hình LSTM với lớp Embedding được học từ đầu cùng với mô hình.

**Code:**
```python
lstm_model = Sequential([
    Embedding(
        input_dim=vocab_size_actual,
        output_dim=100,
        input_length=max_len
    ),
    LSTM(128, dropout=0.5, recurrent_dropout=0),
    Dense(num_classes, activation='softmax')
])
```

**Kết quả:**
-   **Test Accuracy**: 0.018
-   **Test F1-score**: 0.001
-   *Nhận xét*: Mô hình không hội tụ tốt, kết quả tương đương dự đoán ngẫu nhiên.

## 4. Kết luận

Trong bài thực hành này, phương pháp truyền thống **TF-IDF + Logistic Regression** vượt trội hoàn toàn so với các mô hình học sâu (Dense, LSTM). Các nguyên nhân có thể bao gồm:
1.  **Dữ liệu**: Kích thước bộ dữ liệu có thể chưa đủ lớn để huấn luyện các mạng thần kinh sâu phức tạp (LSTM) từ đầu.
2.  **Hyperparameters**: Các tham số huấn luyện (learning rate, batch size, số epoch) của mạng nơ-ron có thể chưa tối ưu.
3.  **Đặc trưng**: Trung bình cộng Word2Vec trong Task 2 có thể làm mất mát quá nhiều thông tin ngữ nghĩa quan trọng so với TF-IDF.

**Hướng cải thiện**:
-   Thử nghiệm tinh chỉnh (fine-tune) embedding pre-trained thay vì đóng băng (freeze).
-   Sử dụng kiến trúc hai chiều (Bi-LSTM) hoặc Attention mechanism.
-   Tăng cường dữ liệu (Data Augmentation) hoặc kiểm tra lại quá trình tiền xử lý dữ liệu.
