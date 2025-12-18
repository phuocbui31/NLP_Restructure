# Báo cáo Lab5 phần 4: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)

## 1. GIỚI THIỆU

### 1.1. Mô tả bài toán
Named Entity Recognition (NER) là một bài toán quan trọng trong xử lý ngôn ngữ tự nhiên, với mục tiêu nhận dạng và phân loại các thực thể có tên trong văn bản như tên người (Person), tên tổ chức (Organization), địa điểm (Location), và các thực thể khác.

### 1.2. Mục tiêu bài lab
Bài lab này xây dựng một mô hình Bidirectional LSTM để giải quyết bài toán NER trên bộ dữ liệu CoNLL 2003, bao gồm:
- Chuẩn bị và tiền xử lý dữ liệu NER
- Xây dựng mô hình Bi-LSTM
- Huấn luyện và đánh giá hiệu năng mô hình

### 1.3. Công nghệ sử dụng
- **Framework**: PyTorch
- **Mô hình**: Bidirectional LSTM (2 layers)
- **Dataset**: CoNLL 2003
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score

---

## 2. TASK 1: CHUẨN BỊ DỮ LIỆU

### 2.1. Bộ dữ liệu CoNLL 2003

#### 2.1.1. Mô tả dataset
- **Nguồn**: CoNLL-2003 Shared Task
- **Ngôn ngữ**: Tiếng Anh
- **Số lượng**: 
  - Training: 14,041 câu
  - Validation: 3,250 câu
  - Test: 3,453 câu

#### 2.1.2. Các nhãn NER (IOB format)
Dataset sử dụng định dạng IOB (Inside-Outside-Beginning):
- `O`: Outside (không phải entity)
- `B-PER`: Beginning of Person
- `I-PER`: Inside Person
- `B-ORG`: Beginning of Organization
- `I-ORG`: Inside Organization
- `B-LOC`: Beginning of Location
- `I-LOC`: Inside Location
- `B-MISC`: Beginning of Miscellaneous
- `I-MISC`: Inside Miscellaneous

**Ví dụ**:
```
Tokens: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
Tags:   ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

### 2.2. Tải dữ liệu

#### 2.2.1. Code implementation
```python
from datasets import load_dataset

# Tải dataset từ Hugging Face
dataset = load_dataset("conll2003", revision="refs/convert/parquet")

# Trích xuất tokens và tags
train_sentences = dataset["train"]["tokens"]
train_tags = dataset["train"]["ner_tags"]
```

#### 2.2.2. Xử lý nhãn
Chuyển đổi nhãn từ dạng số sang string để dễ xử lý:

```python
tag_names = dataset["train"].features["ner_tags"].feature.names

def convert_tags_to_strings(tags_list, tag_names):
    return [[tag_names[tag] for tag in tags] for tags in tags_list]

train_tags_str = convert_tags_to_strings(train_tags, tag_names)
```

### 2.3. Xây dựng Vocabulary

#### 2.3.1. Word Vocabulary
```python
word_to_ix = {"<PAD>": 0, "<UNK>": 1}
for sentence in train_sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
```

**Kết quả**: 23,625 từ duy nhất trong vocabulary

#### 2.3.2. Tag Vocabulary
```python
tag_to_ix = {"<PAD>": 0}
for tag in tag_names:
    if tag not in tag_to_ix:
        tag_to_ix[tag] = len(tag_to_ix)

ix_to_tag = {v: k for k, v in tag_to_ix.items()}
```

**Tag mapping**:
```
{'<PAD>': 0, 'O': 1, 'B-PER': 2, 'I-PER': 3, 'B-ORG': 4, 
 'I-ORG': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-MISC': 8, 'I-MISC': 9}
```

### 2.4. PyTorch Dataset và DataLoader

#### 2.4.1. NERDataset Class
```python
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        
        # Convert to indices
        sentence_indices = torch.tensor(
            [self.word_to_ix.get(word, self.word_to_ix["<UNK>"]) 
             for word in sentence], dtype=torch.long
        )
        tag_indices = torch.tensor(
            [self.tag_to_ix[tag] for tag in tags], dtype=torch.long
        )
        
        return sentence_indices, tag_indices
```

#### 2.4.2. Collate Function (Padding)
```python
def collate_fn(batch):
    sentences, tags = zip(*batch)
    
    # Pad sequences to same length
    sentences_padded = pad_sequence(
        sentences, batch_first=True, 
        padding_value=word_to_ix["<PAD>"]
    )
    tags_padded = pad_sequence(
        tags, batch_first=True, 
        padding_value=tag_to_ix["<PAD>"]
    )
    
    return sentences_padded, tags_padded
```

#### 2.4.3. DataLoader Configuration
```python
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, 
    shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, 
    shuffle=False, collate_fn=collate_fn
)
```

**Kết quả**:
- Train batches: 439
- Validation batches: 102
- Sample batch shape: (32, 43) - 32 câu, mỗi câu 43 tokens (padded)

---

## 3. TASK 2: XÂY DỰNG MÔ HÌNH BI-LSTM

### 3.1. Kiến trúc mô hình

#### 3.1.1. Tổng quan
Mô hình bao gồm 4 thành phần chính:
1. **Embedding Layer**: Chuyển đổi word indices thành dense vectors
2. **Bidirectional LSTM**: Xử lý chuỗi theo cả 2 chiều (forward & backward)
3. **Dropout**: Regularization để tránh overfitting
4. **Linear Layer**: Phân loại tags cho mỗi token

#### 3.1.2. Kiến trúc chi tiết

```
SimpleRNNForTokenClassification(
  (embedding): Embedding(23625, 100, padding_idx=0)
  (rnn): LSTM(100, 128, num_layers=2, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=256, out_features=10, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
)
```

#### 3.1.3. Hyperparameters

| Parameter | Value | Giải thích |
|-----------|-------|------------|
| Vocab size | 23,625 | Số từ trong vocabulary |
| Embedding dim | 100 | Kích thước word embeddings |
| Hidden dim | 128 | Kích thước hidden state của LSTM |
| Num layers | 2 | Số lớp LSTM xếp chồng |
| Bidirectional | True | LSTM xử lý 2 chiều |
| Output size | 10 | Số lượng nhãn NER |
| Dropout | 0.3 | Tỷ lệ dropout |
| **Total params** | **2,995,854** | Tổng số parameters |

### 3.2. Code Implementation

```python
class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 output_size, padding_idx=0, num_layers=1):
        super(SimpleRNNForTokenClassification, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, 
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Output layer (*2 vì bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, sentences):
        # Embedding: (batch, seq_len) -> (batch, seq_len, emb_dim)
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)
        
        # LSTM: (batch, seq_len, emb_dim) -> (batch, seq_len, hidden*2)
        rnn_out, _ = self.rnn(embeds)
        rnn_out = self.dropout(rnn_out)
        
        # Linear: (batch, seq_len, hidden*2) -> (batch, seq_len, output)
        output = self.fc(rnn_out)
        
        return output
```

### 3.3. Ưu điểm của Bi-LSTM

1. **Bidirectional Context**: 
   - Forward LSTM: Học context từ trái sang phải
   - Backward LSTM: Học context từ phải sang trái
   - Kết hợp cả 2 → hiểu được toàn bộ ngữ cảnh câu

2. **Long-term Dependencies**: 
   - LSTM có thể học được quan hệ xa trong câu
   - Tránh được vanishing gradient problem của RNN thông thường

3. **Sequence Labeling**: 
   - Mỗi token được gán nhãn độc lập
   - Phù hợp cho bài toán NER

---

## 4. TASK 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ

### 4.1. Cấu hình huấn luyện

#### 4.1.1. Loss Function
```python
criterion = nn.CrossEntropyLoss(ignore_index=tag_to_ix["<PAD>"])
```
- Sử dụng CrossEntropyLoss cho bài toán multi-class classification
- `ignore_index=0`: Bỏ qua padding tokens khi tính loss

#### 4.1.2. Optimizer
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- Adam optimizer với learning rate 0.001
- Adaptive learning rate cho từng parameter

#### 4.1.3. Learning Rate Scheduler
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)
```
- Giảm learning rate khi validation loss không cải thiện
- Factor: 0.5 (giảm một nửa)
- Patience: 2 epochs

#### 4.1.4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Gradient clipping | max_norm=5.0 |
| Device | CUDA (GPU) |

### 4.2. Quá trình huấn luyện

#### 4.2.1. Training Loop
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for sentences, tags in dataloader:
        sentences, tags = sentences.to(device), tags.to(device)
        
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(sentences)
        
        # 3. Calculate loss
        outputs_flat = outputs.view(-1, outputs.shape[-1])
        tags_flat = tags.view(-1)
        loss = criterion(outputs_flat, tags_flat)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Gradient clipping (tránh exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 6. Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

#### 4.2.2. Kết quả Training

| Epoch | Train Loss | Val Loss | Note |
|-------|------------|----------|------|
| 1 | 0.170 | 0.187 | Saved best model |
| 2 | 0.133 | 0.169 | Saved best model |
| 3 | 0.111 | 0.170 | - |
| 4 | 0.092 | 0.155 | Saved best model |
| 5 | 0.078 | 0.171 | - |

**Best validation loss**: 0.155 (Epoch 4)

#### 4.2.3. Biểu đồ Loss

**Nhận xét**:
- Train loss giảm đều qua các epochs (từ 0.170 → 0.078)
- Val loss tốt nhất ở epoch 4 (0.155)
- Epoch 5 có dấu hiệu overfitting (train giảm, val tăng)
- Model hội tụ tốt sau 4-5 epochs

### 4.3. Đánh giá mô hình

#### 4.3.1. Token-level Accuracy

**Kết quả**: **95.64%** trên tập validation

```python
def calculate_accuracy(model, dataloader, device, tag_to_ix):
    model.eval()
    correct = 0
    total = 0
    pad_idx = tag_to_ix["<PAD>"]
    
    with torch.no_grad():
        for sentences, tags in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            outputs = model(sentences)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Chỉ tính trên non-padding tokens
            mask = (tags != pad_idx)
            correct += ((predictions == tags) & mask).sum().item()
            total += mask.sum().item()
    
    return correct / total
```

#### 4.3.2. Classification Report (Chi tiết theo từng class)

```
              precision    recall  f1-score   support

       B-LOC     0.9158    0.8285    0.8700      1837
      B-MISC     0.8875    0.7614    0.8196       922
       B-ORG     0.8565    0.6898    0.7641      1341
       B-PER     0.7994    0.8480    0.8230      1842
       I-LOC     0.8800    0.7704    0.8216       257
      I-MISC     0.8486    0.6156    0.7136       346
       I-ORG     0.8980    0.7031    0.7886       751
       I-PER     0.8539    0.8944    0.8737      1307
           O     0.9738    0.9893    0.9815     42759

    accuracy                         0.9564     51362
   macro avg     0.8793    0.7889    0.8284     51362
weighted avg     0.9554    0.9564    0.9551     51362
```

#### 4.3.3. Phân tích kết quả

**Các nhãn có F1-score cao:**
- `O` (Outside): 98.15% - Nhận dạng rất tốt các token không phải entity
- `I-PER` (Inside Person): 87.37% - Tốt trong việc nhận dạng phần tiếp theo của tên người
- `B-LOC` (Begin Location): 87.00% - Nhận dạng tốt địa điểm

**Các nhãn có F1-score thấp:**
- `I-MISC` (Inside Miscellaneous): 71.36% - Khó nhất, do MISC là category tổng quát
- `B-ORG` (Begin Organization): 76.41% - Tên tổ chức đa dạng, khó phân biệt
- `I-ORG` (Inside Organization): 78.86% - Tương tự B-ORG

**Nguyên nhân:**
1. **Imbalanced data**: Nhãn `O` chiếm 83% dữ liệu → model thiên về nhãn này
2. **Entity complexity**: ORG và MISC có nhiều biến thể, khó học pattern
3. **Context dependency**: Cần nhiều context hơn để phân biệt các entity tương tự

### 4.4. Test trên câu mới

#### 4.4.1. Hàm predict_sentence

```python
def predict_sentence(sentence, model, word_to_ix, ix_to_tag, device):
    model.eval()
    tokens = sentence.split()
    
    # Convert to indices
    indices = [word_to_ix.get(token, word_to_ix["<UNK>"]) 
               for token in tokens]
    sentence_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(sentence_tensor)
        predictions = torch.argmax(output, dim=-1).squeeze(0)
    
    # Convert back to tags
    predicted_tags = [ix_to_tag[idx.item()] for idx in predictions]
    
    return list(zip(tokens, predicted_tags))
```

#### 4.4.2. Ví dụ dự đoán

**Câu 1**: "VNU University is located in Hanoi"
```
VNU          B-ORG
University   I-ORG
is           O
located      O
in           O
Hanoi        O         ← Sai! Đáng lẽ phải là B-LOC
```

**Câu 2**: "Barack Obama was born in Hawaii"
```
Barack       B-PER     ✓ Đúng!
Obama        I-PER     ✓ Đúng!
was          O         ✓ Đúng!
born         O         ✓ Đúng!
in           O         ✓ Đúng!
Hawaii       B-LOC     ✓ Đúng!
```

**Câu 3**: "Apple Inc. is based in California"
```
Apple        B-ORG     ✓ Đúng!
Inc.         I-ORG     ✓ Đúng!
is           O         ✓ Đúng!
based        O         ✓ Đúng!
in           O         ✓ Đúng!
California   B-LOC     ✓ Đúng!
```

**Câu 4**: "The United Nations met in New York"
```
The          O         ✓ Đúng!
United       B-ORG     ✓ Đúng!
Nations      I-ORG     ✓ Đúng!
met          O         ✓ Đúng!
in           O         ✓ Đúng!
New          B-LOC     ✓ Đúng!
York         I-LOC     ✓ Đúng!
```

**Nhận xét**:
- Model nhận dạng tốt các entity phổ biến (Barack Obama, Apple, United Nations)
- Một số từ mới không có trong training set bị nhận dạng sai (VNU, Hanoi)
- Cần thêm dữ liệu hoặc sử dụng pre-trained embeddings để cải thiện

---

## 5. HƯỚNG DẪN CHẠY CODE

### 5.1. Cài đặt môi trường

#### 5.1.1. Requirements
```bash
pip install torch torchvision
pip install datasets transformers
pip install tqdm matplotlib scikit-learn
```

#### 5.1.2. Kiểm tra GPU
```python
import torch
print(torch.cuda.is_available())  # True nếu có GPU
```

### 5.2. Chạy notebook

#### 5.2.1. Jupyter Notebook
```bash
jupyter notebook lab5_rnn_ner.ipynb
```

#### 5.2.2. Thứ tự thực hiện
1. Chạy cell import thư viện
2. Chạy Task 1: Load và xử lý dữ liệu (~3-5 phút)
3. Chạy Task 2: Tạo Dataset và DataLoader
4. Chạy Task 3: Xây dựng mô hình
5. Chạy Task 4: Training (~5-10 phút với GPU, ~20-30 phút với CPU)
6. Chạy Task 5: Đánh giá và test

### 5.3. Reproduce kết quả

#### 5.3.1. Training từ đầu
```python
# Chạy tất cả cells theo thứ tự
# Model sẽ được lưu tại: best_ner_model.pt
```

#### 5.3.2. Load model đã train
```python
model.load_state_dict(torch.load('best_ner_model.pt'))
```

#### 5.3.3. Test với câu mới
```python
test_sentence = "Your sentence here"
result = predict_sentence(test_sentence, model, word_to_ix, ix_to_tag, device)
```

---

## 6. KHÓ KHĂN VÀ GIẢI PHÁP

### 6.1. Khó khăn gặp phải

#### 6.1.1. Vấn đề load dataset

**Khó khăn**: Lỗi "RuntimeError: Dataset scripts are no longer supported"

**Nguyên nhân**: 
- Hugging Face đã ngừng hỗ trợ dataset scripts từ phiên bản datasets >= 2.15.0
- Dataset CoNLL 2003 cũ vẫn dùng Python script

**Giải pháp**:
```python
# Thay vì:
dataset = load_dataset("conll2003", trust_remote_code=True)

# Sử dụng:
dataset = load_dataset("conll2003", revision="refs/convert/parquet")
```

#### 6.1.2. Padding và masking

**Khó khăn**: Loss tính cả trên padding tokens, làm sai kết quả

**Giải pháp**:
```python
# Sử dụng ignore_index trong loss function
criterion = nn.CrossEntropyLoss(ignore_index=tag_to_ix["<PAD>"])

# Và mask khi tính accuracy
mask = (tags != pad_idx)
correct = ((predictions == tags) & mask).sum()
```

#### 6.1.3. Exploding gradients

**Khó khăn**: Gradients quá lớn trong LSTM, gây loss = NaN

**Giải pháp**:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

#### 6.1.4. Overfitting

**Khó khăn**: Model overfit sau epoch 4-5

**Giải pháp**:
- Dropout = 0.3
- Learning rate scheduler
- Early stopping (save best model based on validation loss)

### 6.2. Các vấn đề kỹ thuật khác

#### 6.2.1. Memory issues với batch size lớn

**Giải pháp**: Giảm batch size từ 64 → 32

#### 6.2.2. Training chậm

**Giải pháp**: 
- Sử dụng GPU (CUDA)
- Giảm số epochs nếu cần kết quả nhanh

#### 6.2.3. Unknown words (UNK)

**Giải pháp**: Thêm token `<UNK>` vào vocabulary và xử lý trong Dataset

---

## 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Kết quả đạt được

- Xây dựng được pipeline hoàn chỉnh cho bài toán NER
- Đạt accuracy 95.64% trên validation set
- F1-score weighted average: 95.51%
- Model nhận dạng tốt các entity phổ biến

### 7.2. Hạn chế

- F1-score thấp cho một số class (MISC: 71.36%, ORG: 76.41%)
- Không nhận dạng tốt từ mới (out-of-vocabulary)
- Chưa xử lý được các entity có nhiều biến thể
- Imbalanced data (O chiếm 83%)

### 7.3. Hướng cải thiện

#### 7.3.1. Về mô hình
1. **Thêm CRF layer**: Bi-LSTM-CRF để tận dụng transition probabilities
2. **Pre-trained embeddings**: GloVe, Word2Vec, hoặc BERT embeddings
3. **Character-level features**: Học morphology của từ
4. **Ensemble methods**: Kết hợp nhiều models

#### 7.3.2. Về dữ liệu
1. **Data augmentation**: Tăng cường dữ liệu cho class ít
2. **Class weighting**: Đặt trọng số cao hơn cho class ít
3. **External data**: Thêm dữ liệu từ nguồn khác

#### 7.3.3. Về huấn luyện
1. **Hyperparameter tuning**: Grid search, Random search
2. **Different optimizers**: SGD with momentum, AdamW
3. **Longer training**: Tăng số epochs với early stopping

---

## 8. TÀI LIỆU THAM KHẢO

1. CoNLL 2003 Dataset: https://huggingface.co/datasets/conll2003
2. PyTorch LSTM Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
3. Named Entity Recognition Paper: https://arxiv.org/abs/1603.01360
4. Bidirectional LSTM-CRF Models for Sequence Tagging: https://arxiv.org/abs/1508.01991
5. Tài liệu giảng dạy trên lớp


### Files trong project

```
Lab5/part4/
├── lab5_rnn_ner.ipynb           # Notebook chính
├── README.md                     # Báo cáo (file này)
├── best_ner_model.pt            # Model weights (sau khi train)
└── ner_model_checkpoint.pt      # Full checkpoint (sau khi train)
```
