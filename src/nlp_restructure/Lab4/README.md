# Báo cáo Lab4: Text Classification & Sentiment Analysis

## 1. Giải thích chi tiết các bước triển khai

### Task 1. Scikit-learn TextClassifier

*   Xây dựng class `TextClassifier` trong `src/models/text_classifier.py` với các phương thức:

    *   `fit`: Huấn luyện mô hình Logistic Regression trên dữ liệu văn bản đầu vào đã vector hóa.
    *   `predict`: Dự đoán cho tập dữ liệu mới.
    *   `evaluate`: Tính toán các chỉ số đánh giá (Accuracy, Precision, Recall, F1-score).

### Task 2. Evaluation

*   Tạo file `test/lab5_test.py` với các nhiệm vụ:

    *   Chia tập train/test.
    *   Tiền xử lý văn bản bằng RegexTokenizer và CountVectorizer.
    *   Huấn luyện, dự đoán và đánh giá mô hình TextClassifier trên dữ liệu có sẵn.

### Task 3. Sentiment Analysis with PySpark

*   Chạy script test/lab5_spark_sentiment_analysis.py:
    *   Đọc dữ liệu cảm xúc từ file CSV.
    *   Tiền xử lý: chuẩn hóa nhãn, loại bỏ NA, chia tập train/test.
    *   Xây dựng pipeline Spark ML gồm các bước: Tokenizer, StopWordsRemover, HashingTF, IDF, LogisticRegression.
    *   Huấn luyện và đánh giá mô hình trên tập dữ liệu với Spark.

### Task 4. Evaluating and Improving Model Performance

Thực hiện phương pháp để cải thiện chất lượng của mô hình.

## 2. Hướng dẫn cách chạy code

*   Đảm bảo cài đặt thư viện uv. Nếu chưa cài đặt, có thể sử dụng lệnh: ```curl -LsSf https://astral.sh/uv/install.sh | sh```

*   Khởi tạo môi trường để chạy code:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

*   Chạy test các task trên bằng lệnh:
    ```bash
    # Task 2: Evaluation classifier with scikit-learn
    uv run -m Lab4.test.lab5_test

    # Task 3: Run Sentiment Analysis with PySpark
    uv run -m Lab4.test.lab5_spark_sentiment_analysis # Pipeline Spark ML cơ bản với LogisticRegression

    uv run -m Lab4.test.lab5_spark_sentiment_analysis_approach1 # Cải thiện Pipeline Spark ML với preprocessing (lọc nhiễu, giảm chiều vector từ vựng, giảm chiều của vector đặc trưng)

    # Task 4: Improving Model Performance
    uv run -m Lab4.test.lab5_spark_sentiment_analysis_advanced # Pipeline kết hợp các cải tiến: preprocessing, embedding, các model như Logistic Regression, Naive Bayes, GBT, Neural Net
    ```

Các kết quả của từng task được lưu tại thư mục Lab4/results/ + tên file tương ứng.

## 3. Phân tích kết quả

### Task 2: Thử nghiệm cơ bản (với 6 mẫu dữ liệu)

*   Kết quả:

    *   Thời gian huấn luyện/dự đoán: Gần như bằng 0 (0.007s / 0.000s).

    *   Metrics (Accuracy, F1, v.v.): 0.000.

*   Phân tích:

    *   Mô hình thất bại hoàn toàn (độ chính xác 0%).
    *  Nguyên nhân là do bộ dữ liệu quá nhỏ (chỉ 6 mẫu). Khi chia train/test (5/1), mẫu kiểm tra duy nhất chứa các từ ("superb", "acting") không có trong tập huấn luyện.
    *   CountVectorizer tạo ra một vector toàn số 0, và mô hình dự đoán sai.

### Task 3: Pipeline Spark ML cơ bản

*   Kết quả:

    *   Thời gian: Huấn luyện (5.38s), Đánh giá (1.38s).

    *   Metrics: Accuracy (0.7295), F1 (0.7266).

*   Phân tích:

    *   Đây là mô hình cơ sở (baseline) của bạn trên một bộ dữ liệu thực tế.

    *   Mô hình đã học được (Accuracy khoảng 73%), tốt hơn nhiều so với việc đoán ngẫu nhiên.

### Task 3 (Cải thiện): Pipeline với Preprocessing

*   Kết quả:

    *   Thời gian: Huấn luyện (7.78s), Đánh giá (0.13s).

    *   Metrics: Accuracy (0.742), F1 (0.742).

*   Phân tích:

    *   Hiệu suất mô hình: Cả Accuracy và F1 đều tăng lên (từ khoảng 73% lên 74.2%).

    *   Thời gian huấn luyện: Tăng lên (từ 5.38s lên 7.78s). Điều này là hợp lý vì phương pháp này đã thêm các bước tiền xử lý (lọc nhiễu, thống kê tần suất từ) trước khi huấn luyện.

    *   Thời gian đánh giá: Giảm mạnh (từ 1.38s xuống 0.13s). Điều này là do giảm chiều vector đặc trưng (từ 10000 xuống 3000), khiến việc dự đoán trên dữ liệu mới nhanh hơn.

### Task 4: So sánh nhiều mô hình và Embedding

*   Kết quả:

    *   Các mô hình dùng tfidf_features (LR, NaiveBayes, NeuralNet) đều cho kết quả tốt hơn (F1: 0.71 đến 0.74) so với các mô hình dùng w2v_features (F1: 0.50 đến 0.62).

    *   Mô hình tốt nhất: NeuralNet (với TF-IDF) cho kết quả cao nhất (F1: 0.743), gần như tương đương với Task 3 Cải thiện. NaiveBayes cũng là một lựa chọn tốt, cho kết quả khá (F1: 0.717) với thời gian huấn luyện nhanh (9.18s).

*   Phân tích:

    *   TF-IDF vs. Word2Vec: Đối với bài toán phân tích cảm xúc (văn bản ngắn, tín hiệu mạnh), việc một từ xuất hiện hay không (như "terrible", "fantastic") thường quan trọng hơn ngữ nghĩa bối cảnh của nó. TF-IDF nắm bắt tín hiệu này rất tốt. Word2Vec (đặc biệt khi chỉ lấy trung bình vector) có thể làm "loãng" các tín hiệu cảm xúc mạnh này.

    *   Model Selection: NeuralNet và LogisticRegression hoạt động tốt trên dữ liệu TF-IDF (vốn thưa - sparse). GBT (mô hình dựa trên cây) thường hoạt động kém hơn trên dữ liệu văn bản thưa và cũng tốn thời gian huấn luyện nhất (với 26.27s).

## 4. Khó khăn thực tế và giải pháp
*   Xử lý label: Dữ liệu gốc có label -1, 1. Phải chuyển -1 thành 0 để phù hợp với các mô hình Spark ML (yêu cầu label là số nguyên không âm).
*   Chất lượng embedding: Word2Vec cần dữ liệu lớn và đa dạng để học embedding tốt. Có thể thử pre-trained embedding (GloVe, FastText) nếu muốn cải thiện.
*   Tuning hyperparameter: Việc chọn tham số tối ưu cho các mô hình (như learning rate, số chiều embedding, số layer NeuralNet, maxIter...) ảnh hưởng lớn đến kết quả. Cần thử nghiệm grid search hoặc random search để tìm cấu hình tốt nhất.
*   Reproducibility: Kết quả có thể thay đổi giữa các lần chạy do random seed, chia dữ liệu train/test khác nhau. Nên cố định seed và ghi rõ quy trình chia dữ liệu để đảm bảo tái lập.
*   Dữ liệu thiếu hoặc lỗi: Một số dòng dữ liệu có thể bị thiếu trường hoặc lỗi định dạng, cần kiểm tra và loại bỏ kỹ trước khi huấn luyện.
*   Triển khai thực tế: Khi áp dụng trên dữ liệu lớn hơn hoặc môi trường production, cần cân nhắc về tài nguyên tính toán, thời gian huấn luyện, và khả năng mở rộng (scaling) của pipeline
