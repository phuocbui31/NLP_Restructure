# Báo cáo Bài tập Lab2 - Spark NLP Pipeline

Mục tiêu: Xây dựng pipeline tiền xử lý văn bản và tạo biểu diễn TF‑IDF bằng Spark ML (Tokenizer → StopWordsRemover → HashingTF → IDF) trên mẫu dữ liệu c4-train.00000-of-01024.json.gz.

---

## 1) Các bước triển khai
- Môi trường:
  - Cài đặt jdk và sbt:
    - `sudo apt update`
    - `sudo apt install openjdk-17-jdk`
    - `echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list`
    - `sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823`
    - `sudo apt update`
    - `sudo apt install sbt`
- Mã nguồn chính:
  - spark_labs/src/main/scala/... chứa lớp chính com.phuocbui.spark.Lab17_NLPPipeline.
  - build.sbt nằm ở spark_labs/.
- Dữ liệu:
  - C4 Common Crawl dataset (30K records)
  - Vị trí lưu trữ: /home/phuocbui3102/NLP/Lab2/data/c4-train.00000-of-01024.json.gz
- Pipeline thực thi:
  1. Đọc JSON: spark.read.json(...).limit(1000) để lấy sample kiểm thử.
  2. Tokenize: Tokenizer hoặc RegexTokenizer trên trường text.
  3. Loại stop words: StopWordsRemover.
  4. Biểu diễn tần suất: HashingTF (setNumFeatures = 20000).
  5. Chuyển sang TF‑IDF: IDF.fit() → transform().
  6. Ghi metrics và kết quả mẫu ra file.

---

## 2) Cách chạy và ghi log kết quả
- Chạy bằng sbt (từ thư mục spark_labs):
  ```
  cd /home/phuocbui3102/NLP/Lab2/spark_labs
  sbt run
  ```

- File log và kết quả:
  - Metrics: /home/phuocbui3102/NLP/Lab2/spark_labs/log/lab17_metrics.log.
  - Results: /home/phuocbui3102/NLP/Lab2/spark_labs/results/lab17_pipeline_output.txt

- Mở Spark UI khi chạy: http://localhost:4040

---

## 3) Giải thích kết quả thu được
- Metrics (lab17_metrics.log) gồm:
  - Thời gian fitting pipeline 3.62 giây.
  - Thời gian transform dữ liệu 1.52 giây.
  - Kích thước vocabulary thực tế sau tokenization và loại stop word: 46838
  - Thông báo nếu numFeatures (20000) nhỏ hơn vocabulary.

- Ý nghĩa:
  - TF‑IDF biểu diễn tầm quan trọng từ theo document, dùng cho clustering/classification.
  - HashingTF cố định kích thước vector; nếu vocabulary lớn hơn numFeatures → collisions làm mất thông tin.

---

## 4) Khó khăn gặp phải và cách khắc phục
- Đọc file .json.gz lớn, chậm/tốn RAM:
  - Dùng .limit(...) khi thử nghiệm; tăng memory driver/executor; xử lý theo partitions.
- Hash collisions (numFeatures quá nhỏ):
  - Tăng setNumFeatures hoặc dùng CountVectorizer để xây vocab trước.
- Thời gian fit/transform lớn:
  - Chạy trên cluster, dùng local[*] với nhiều core, cache() DataFrame trung gian, tune partitions.
- Vấn đề dependency (sbt/scala/spark):
  - Đồng bộ phiên bản Scala ↔ Spark (thư mục target cho thấy scala-2.12); dùng sbt clean/update.
- Spark UI/port 4040 không hiển thị:
  - Kiểm tra process, xung đột cổng, xem logs.

---

## 5) Tham khảo
- Apache Spark MLlib — Feature extraction (Tokenizer, StopWordsRemover, HashingTF, IDF): https://spark.apache.org/docs/latest/ml-features.html

- C4 / Common Crawl (dữ liệu): https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz

---

## 6) Mô hình tiền huấn luyện
- Không sử dụng mô hình tiền huấn luyện trong Lab2. Chỉ dùng các transformer tiêu chuẩn của Spark ML (Tokenizer, StopWordsRemover, HashingTF, IDF).

---

Tệp liên quan nhanh:
- Code chính: spark_labs/src/main/scala/... (com.phuocbui.spark.Lab17_NLPPipeline)
- Build: spark_labs/build.sbt
- Dữ liệu mẫu: data/c4-train.00000-of-01024.json.gz
- Metrics log: spark_labs/log/lab17_metrics.log
- Results: spark_labs/results/lab17_pipeline_output.txt
