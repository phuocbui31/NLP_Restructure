# Lab 3 part 1: Word Embeddings, Visualization & Analysis

## 1. Source code, dữ liệu, kết quả sử dụng

- **Notebook**: [notebook/Lab3/22001630_Lab3_Word_Embedding.pdf](../../notebook/Lab3/22001630_Lab3_Word_Embedding.pdf) (Báo cáo chi tiết dạng PDF)

### 1.1. Dataset sử dụng

- **Tên dataset**: GloVe Pre-trained Word Vectors
- **Mô tả**: Pre-trained word embeddings được huấn luyện trên 6 tỷ tokens từ Wikipedia 2014 + Gigaword 5.
- **Cấu trúc dữ liệu**:
  - **Format**: Text file, mỗi dòng gồm: `word` + `vector values` (space-separated).
  - **Số lượng**: 400,000 từ vựng.
  - **Kích thước vector**: 50d, 100d, 200d, 300d (sử dụng phiên bản **100d**).
  - **Kiểu dữ liệu**: String (word) + Float values (vector).
- **Nguồn**: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- **File sử dụng**: `glove.6B.100d.txt` (~331 MB).

## 2. Các bước thực hiện

1. **Load Pre-trained Vectors**: Sử dụng thư viện Gensim để load GloVe vectors.
2. **Dimensionality Reduction**: Giảm chiều vector từ 100D xuống 2D bằng thuật toán PCA (Principal Component Analysis).
3. **Visualization**: Trực quan hóa các từ trong không gian 2D sử dụng thư viện `matplotlib`.
4. **Similarity Search**: Tìm kiếm Top K từ tương đồng với một từ bất kỳ (ví dụ: "king") dựa trên cosine similarity.
5. **Analysis**: Hiển thị kết quả, phân tích các cụm từ và độ tương đồng semantic.

## 3. Hướng dẫn chạy code

1. Mở notebook `lab3_word_embeddings.ipynb` bằng Jupyter Notebook hoặc Google Colab.
2. Đảm bảo file GloVe vectors (`glove.6B.100d.txt`) đã được tải và giải nén vào thư mục `data/`.
3. Chạy tuần tự các cell để thực hiện load model, tính toán PCA và hiển thị hình ảnh trực quan hóa.

## 4. Nhận xét về độ tương đồng và các từ đồng nghĩa

- Các từ đồng nghĩa/tương đồng tìm được từ model pre-trained GloVe rất hợp lý về mặt ngữ nghĩa:
  - Ví dụ: `computer` → `computers`, `software`, `technology`, `pc`, `hardware`.
- Độ tương đồng cosine (cosine similarity) cao cho thấy model đã học tốt các mối quan hệ ngữ nghĩa giữa các từ trong tập dữ liệu huấn luyện khổng lồ.

## 5. Phân tích biểu đồ trực quan hóa

- **Clustering**: Các từ có ý nghĩa liên quan thường được nhóm lại gần nhau trong không gian vector:
  - Cụm từ hoàng gia: `king`, `queen`, `prince`, `monarch`, `kingdom`.
  - Cụm từ công nghệ: `computer`, `software`, `technology`.
  - Cụm từ địa lý: `country`, `city`, `state`.
- **Giải thích**: GloVe học các vector từ dựa trên thống kê đồng xuất hiện (co-occurrence matrix). Thuật toán PCA khi giảm chiều dữ liệu cố gắng bảo toàn phương sai (variance) lớn nhất, do đó các khoảng cách tương đối giữa các cụm từ vẫn được giữ lại một cách đáng kể.

## 6. So sánh model pre-trained và model tự huấn luyện

- **Pre-trained GloVe**: Có chất lượng tốt nhất do được huấn luyện trên lượng dữ liệu text khổng lồ (Wikipedia + Gigaword), vocabulary phong phú và similarity scores chính xác.
- **Model tự huấn luyện (Spark Word2Vec)**: Vocabulary thường nhỏ hơn (phụ thuộc vào dataset training như UD_English-EWT), chất lượng semantic ở mức vừa phải nhưng đủ để học các mối quan hệ cơ bản trong domain cụ thể. huấn luyện trên dataset lớn như C4 (Lab 2) sẽ cho kết quả tốt hơn.

## 7. Khó khăn và giải pháp

- **Xử lý file lớn**: File GloVe vectors và model embedding tốn nhiều RAM.
  - *Giải pháp*: Load từng phần hoặc dùng `gensim` với mmap, hoặc chỉ load top N từ phổ biến.
- **PCA Visualization**: Chạy PCA trên toàn bộ 400k từ rất tốn thời gian và biểu đồ sẽ bị rối (cluttered).
  - *Giải pháp*: Chỉ lấy subset (ví dụ: 100-200 từ) tiêu biểu hoặc các từ muốn phân tích để visualize.
- **Format khác biệt**: Cần chuyển đổi format vector giữa text thuần, Gensim binary, và Spark vector.
  - *Giải pháp*: Viết các wrapper class hoặc script convert định dạng chuẩn.

## 8. Nguồn tham khảo

- [Stanford GloVe Project](https://nlp.stanford.edu/projects/glove/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Tài liệu Word Embedding trên lớp
