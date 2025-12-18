# Báo cáo Lab5 phần 3: Xây dựng mô hình RNN cho bài toán Gán nhãn Từ loại (POS Tagging)

## 1. GIỚI THIỆU

### 1.1. Mô tả bài toán
Part-of-Speech (POS) Tagging là một bài toán quan trọng trong xử lý ngôn ngữ tự nhiên, với mục tiêu gán nhãn từ loại cho mỗi từ trong câu. Các từ loại bao gồm: danh từ (NOUN), động từ (VERB), tính từ (ADJ), trạng từ (ADV), đại từ (PRON), và nhiều loại khác.

### 1.2. Mục tiêu bài lab
Bài lab này xây dựng một mô hình RNN để giải quyết bài toán POS Tagging trên bộ dữ liệu Universal Dependencies, bao gồm:
- Chuẩn bị và tiền xử lý dữ liệu POS Tagging
- Xây dựng mô hình RNN (với hỗ trợ LSTM, GRU)
- Huấn luyện và đánh giá hiệu năng mô hình

### 1.3. Công nghệ sử dụng
- **Framework**: PyTorch
- **Mô hình**: Simple RNN / LSTM / GRU
- **Dataset**: Universal Dependencies (UD English EWT)
- **Evaluation metrics**: Accuracy

---

## 2. TASK 1: CHUẨN BỊ DỮ LIỆU

### 2.1. Bộ dữ liệu Universal Dependencies

#### 2.1.1. Mô tả dataset
- **Nguồn**: Universal Dependencies - English EWT
- **Ngôn ngữ**: Tiếng Anh
- **Số lượng**: 
  - Training: 12,543 câu
  - Validation: 2,002 câu
  - Test: 2,077 câu

#### 2.1.2. Các nhãn POS (UPOS format)
Dataset sử dụng định dạng Universal POS tags:
- `NOUN`: Danh từ
- `VERB`: Động từ
- `ADJ`: Tính từ
- `ADV`: Trạng từ
- `PRON`: Đại từ
- `DET`: Mạo từ/Định từ
- `ADP`: Giới từ
- `CCONJ`: Liên từ đẳng lập
- `PROPN`: Danh từ riêng
- `AUX`: Trợ động từ
- Và các nhãn khác...

**Ví dụ**:
```
Tokens: ['I', 'love', 'NLP']
Tags:   ['PRON', 'VERB', 'PROPN']
```

### 2.2. Tải dữ liệu

#### 2.2.1. Code implementation
```python
from datasets import load_dataset

# Tải dataset từ Hugging Face
dataset = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)

# Trích xuất dữ liệu
train_sentences = [(tokens, tags) for tokens, tags in 
                   zip(dataset['train']['tokens'], dataset['train']['upos'])]
```

#### 2.2.2. Cấu trúc dữ liệu
Mỗi câu được biểu diễn dưới dạng danh sách các cặp (word, tag):
```python
# Ví dụ một câu
[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN')]
```

### 2.3. Xây dựng Vocabulary

#### 2.3.1. Build Vocabulary Function
```python
from collections import Counter

def build_vocab(sentences, min_freq=2):
    word_counter = Counter()
    tag_counter = Counter()
    
    for sentence in sentences:
        for word, tag in sentence:
            word_counter[word] += 1
            tag_counter[tag] += 1
    
    # Tag vocabulary
    tag_to_ix = {'<PAD>': 0}
    for tag in tag_counter.keys():
        tag_to_ix[tag] = len(tag_to_ix)
    
    # Word vocabulary với min_freq filter
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.items():
        if count >= min_freq:
            word_to_ix[word] = len(word_to_ix)
    
    return word_to_ix, tag_to_ix

word_to_ix, tag_to_ix = build_vocab(train_sentences, min_freq=2)
ix_to_tag = {v: k for k, v in tag_to_ix.items()}
```

**Kết quả**:
- Vocabulary size: ~15,000 từ duy nhất
- Tag vocabulary: 18 tags (bao gồm `<PAD>`)

### 2.4. PyTorch Dataset và DataLoader

#### 2.4.1. POSDataset Class
```python
class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix, lowercase=False):
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.lowercase = lowercase
        
        self.unk_idx = word_to_ix.get('<UNK>', 1)
        self.pad_tag_idx = tag_to_ix.get('<PAD>', 0)
        
        # Pre-convert all sentences to tensors
        self.data = []
        for sentence in sentences:
            word_indices = []
            tag_indices = []
            for word, tag in sentence:
                w = word.lower() if self.lowercase else word
                word_idx = self.word_to_ix.get(w, self.unk_idx)
                tag_idx = self.tag_to_ix.get(tag, self.pad_tag_idx)
                word_indices.append(word_idx)
                tag_indices.append(tag_idx)
            
            self.data.append((
                torch.LongTensor(word_indices), 
                torch.LongTensor(tag_indices)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

#### 2.4.2. Collate Function (Padding)
```python
def collate_fn(batch):
    sentences, tags = zip(*batch)
    
    # Tính độ dài của mỗi câu
    lengths = torch.LongTensor([s.size(0) for s in sentences])
    
    # Pad sequences to same length
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    
    return padded_sentences, padded_tags, lengths
```

#### 2.4.3. DataLoader Configuration
```python
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, 
    shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_dataset, batch_size=BATCH_SIZE, 
    shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, 
    shuffle=False, collate_fn=collate_fn
)
```

**Kết quả**:
- Train batches: 392
- Validation batches: 63
- Test batches: 65

---

## 3. TASK 2: XÂY DỰNG MÔ HÌNH RNN

### 3.1. Kiến trúc mô hình

#### 3.1.1. Tổng quan
Mô hình bao gồm 3 thành phần chính:
1. **Embedding Layer**: Chuyển đổi word indices thành dense vectors
2. **RNN Layer**: Xử lý chuỗi tuần tự (hỗ trợ RNN, LSTM, GRU)
3. **Linear Layer**: Phân loại tags cho mỗi token

#### 3.1.2. Kiến trúc chi tiết

```
SimpleRNNForTokenClassification(
  (embedding): Embedding(vocab_size, 128, padding_idx=0)
  (rnn): RNN(128, 256, batch_first=True)
  (fc): Linear(in_features=256, out_features=18, bias=True)
)
```

#### 3.1.3. Hyperparameters

| Parameter | Value | Giải thích |
|-----------|-------|------------|
| Vocab size | ~15,000 | Số từ trong vocabulary |
| Embedding dim | 128 | Kích thước word embeddings |
| Hidden dim | 256 | Kích thước hidden state của RNN |
| Num layers | 1 | Số lớp RNN xếp chồng |
| Output size | 18 | Số lượng nhãn POS |

### 3.2. Code Implementation

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size,
                 num_layers=1, bidirectional=False, dropout=0.0, rnn_type='RNN'):
        super(SimpleRNNForTokenClassification, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer (hỗ trợ RNN, LSTM, GRU)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}.get(rnn_type, nn.RNN)
        
        self.rnn = rnn_cls(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, tagset_size)
    
    def forward(self, sentences, lengths=None):
        embeds = self.embedding(sentences)
        
        # Sử dụng pack_padded_sequence nếu có lengths
        if lengths is not None:
            packed = pack_padded_sequence(
                embeds, lengths.cpu(), 
                batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, padding_value=0.0
            )
        else:
            rnn_out, _ = self.rnn(embeds)
        
        tag_scores = self.fc(rnn_out)
        return tag_scores
```

### 3.3. Ưu điểm của kiến trúc

1. **Flexible RNN Type**: 
   - Hỗ trợ cả RNN, LSTM, và GRU
   - Dễ dàng thử nghiệm các loại RNN khác nhau

2. **Packed Sequences**: 
   - Sử dụng `pack_padded_sequence` để tối ưu hóa computation
   - Bỏ qua padding tokens trong quá trình forward pass

3. **Bidirectional Support**: 
   - Có thể bật/tắt bidirectional processing
   - Tăng cường khả năng học context

---

## 4. TASK 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ

### 4.1. Cấu hình huấn luyện

#### 4.1.1. Loss Function
```python
criterion = nn.CrossEntropyLoss(ignore_index=0)
```
- Sử dụng CrossEntropyLoss cho bài toán multi-class classification
- `ignore_index=0`: Bỏ qua padding tokens khi tính loss

#### 4.1.2. Optimizer
```python
LEARNING_RATE = 0.001
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

#### 4.1.3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Device | CUDA (GPU) |

### 4.2. Quá trình huấn luyện

#### 4.2.1. Training Loop
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for sentences, tags, lengths in tqdm(dataloader, desc="Training"):
        sentences = sentences.to(device)
        tags = tags.to(device)
        
        optimizer.zero_grad()
        
        tag_scores = model(sentences)
        
        # Flatten for loss calculation
        tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
        tags = tags.view(-1)
        
        loss = criterion(tag_scores, tags)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss
```

#### 4.2.2. Kết quả Training

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1 | 0.969 | 0.677 | 78.18% |
| 2 | 0.551 | 0.521 | 83.13% |
| 3 | 0.405 | 0.449 | 84.95% |
| 4 | 0.316 | 0.411 | 86.23% |
| 5 | 0.257 | 0.393 | 87.47% |
| 6 | 0.214 | 0.386 | 87.44% |
| 7 | 0.181 | 0.391 | 87.31% |
| 8 | 0.155 | 0.388 | **88.10%** |
| 9 | 0.133 | 0.401 | 88.04% |
| 10 | 0.114 | 0.417 | 87.60% |

**Best validation accuracy**: 88.10% (Epoch 8)

#### 4.2.3. Nhận xét về quá trình training
- Train loss giảm đều qua các epochs (từ 0.969 → 0.114)
- Val accuracy tốt nhất ở epoch 8 (88.10%)
- Sau epoch 8, có dấu hiệu overfitting (val loss tăng, val accuracy giảm nhẹ)
- Model hội tụ tốt sau 8-10 epochs

### 4.3. Đánh giá mô hình

#### 4.3.1. Kết quả trên tập Test

```
Result on test data
Loss: 0.415
Accuracy: 87.515%
```

#### 4.3.2. Evaluation Function
```python
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sentences, tags, lengths in tqdm(dataloader, desc="Evaluating"):
            sentences = sentences.to(device)
            tags = tags.to(device)
            
            tag_scores = model(sentences)
            
            # Calculate loss
            tag_scores_flat = tag_scores.view(-1, tag_scores.shape[-1])
            tags_flat = tags.view(-1)
            loss = criterion(tag_scores_flat, tags_flat)
            total_loss += loss.item()
            
            # Calculate accuracy (ignoring padding)
            predictions = torch.argmax(tag_scores, dim=-1)
            mask = (tags != 0)
            correct += ((predictions == tags) & mask).sum().item()
            total += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
```

### 4.4. Test trên câu mới

#### 4.4.1. Hàm predict_sentence

```python
def predict_sentence(sentence, model, word_to_ix, ix_to_tag, device):
    model.eval()
    
    # Tokenize
    words = sentence.split()
    
    # Convert to indices
    unk_idx = word_to_ix['<UNK>']
    word_indices = [word_to_ix.get(word, unk_idx) for word in words]
    
    # Convert to tensor
    sentence_tensor = torch.LongTensor(word_indices).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        tag_scores = model(sentence_tensor)
        predictions = torch.argmax(tag_scores, dim=-1).squeeze(0)
    
    # Convert predictions to tags
    predicted_tags = [ix_to_tag[idx.item()] for idx in predictions]
    
    return list(zip(words, predicted_tags))
```

#### 4.4.2. Ví dụ dự đoán

**Câu 1**: "I love NLP"
```
I                    -> PRON    ✓ Đúng!
love                 -> VERB    ✓ Đúng!
NLP                  -> PROPN   ✓ Đúng!
```

**Câu 2**: "She quickly ran to the station building"
```
She                  -> PRON    ✓ Đúng!
quickly              -> ADV     ✓ Đúng!
ran                  -> VERB    ✓ Đúng!
to                   -> ADP     ✓ Đúng!
the                  -> DET     ✓ Đúng!
station              -> NOUN    ✓ Đúng!
building             -> NOUN    ✓ Đúng!
```

**Câu 3**: "Deep learning models are powerful and effective"
```
Deep                 -> NOUN    (Có thể là ADJ)
learning             -> VERB    (Có thể là NOUN/ADJ)
models               -> NOUN    ✓ Đúng!
are                  -> AUX     ✓ Đúng!
powerful             -> ADJ     ✓ Đúng!
and                  -> CCONJ   ✓ Đúng!
effective            -> ADJ     ✓ Đúng!
```

**Nhận xét**:
- Model nhận dạng tốt các từ thông dụng
- Một số từ có nhiều nghĩa (ambiguous) có thể bị gán nhãn sai
- Từ mới không có trong training set sẽ được xử lý như `<UNK>`

---

## 5. HƯỚNG DẪN CHẠY CODE

### 5.1. Cài đặt môi trường

#### 5.1.1. Requirements
```bash
pip install torch torchvision
pip install datasets transformers
pip install tqdm matplotlib
```

#### 5.1.2. Kiểm tra GPU
```python
import torch
print(torch.cuda.is_available())  # True nếu có GPU
```

### 5.2. Chạy notebook

#### 5.2.1. Jupyter Notebook
```bash
jupyter notebook Lab5_RNN_for_pos_tagging.ipynb
```

#### 5.2.2. Thứ tự thực hiện
1. Chạy cell import thư viện
2. Chạy Task 1: Load và xử lý dữ liệu
3. Chạy Task 2: Tạo Dataset và DataLoader
4. Chạy Task 3: Xây dựng mô hình
5. Chạy Task 4: Training (~5-10 phút với GPU)
6. Chạy Task 5: Đánh giá và test

### 5.3. Reproduce kết quả

#### 5.3.1. Load model đã train
```python
checkpoint = torch.load('RNN_pos_tagging_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
word_to_ix = checkpoint['word_to_ix']
ix_to_tag = checkpoint['ix_to_tag']
```

#### 5.3.2. Test với câu mới
```python
test_sentence = "Your sentence here"
result = predict_sentence(test_sentence, model, word_to_ix, ix_to_tag, device)
for word, tag in result:
    print(f"{word:20s} -> {tag}")
```

---

## 6. KHÓ KHĂN VÀ GIẢI PHÁP

### 6.1. Khó khăn gặp phải

#### 6.1.1. Variable length sequences

**Khó khăn**: Các câu có độ dài khác nhau, cần padding để batch processing

**Giải pháp**:
```python
# Sử dụng pack_padded_sequence và pad_packed_sequence
packed = pack_padded_sequence(embeds, lengths.cpu(), 
                               batch_first=True, enforce_sorted=False)
packed_out, _ = self.rnn(packed)
rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
```

#### 6.1.2. Out-of-Vocabulary (OOV) words

**Khó khăn**: Từ mới không có trong vocabulary

**Giải pháp**:
```python
# Sử dụng <UNK> token và min_freq filtering
word_to_ix = {'<PAD>': 0, '<UNK>': 1}
for word, count in word_counter.items():
    if count >= min_freq:
        word_to_ix[word] = len(word_to_ix)
```

#### 6.1.3. Padding và masking

**Khó khăn**: Loss tính cả trên padding tokens

**Giải pháp**:
```python
# Sử dụng ignore_index trong loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Và mask khi tính accuracy
mask = (tags != 0)
correct = ((predictions == tags) & mask).sum()
```

### 6.2. So sánh các loại RNN

| RNN Type | Ưu điểm | Nhược điểm |
|----------|---------|------------|
| SimpleRNN | Nhanh, ít parameters | Vanishing gradient |
| LSTM | Long-term dependencies | Nhiều parameters hơn |
| GRU | Balance giữa RNN và LSTM | Có thể kém LSTM |

---

## 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Kết quả đạt được

- Xây dựng được pipeline hoàn chỉnh cho bài toán POS Tagging
- Đạt accuracy **87.5%** trên tập test
- Model nhận dạng tốt các từ loại phổ biến
- Code flexible, hỗ trợ nhiều loại RNN

### 7.2. Hạn chế

- Accuracy chưa cao bằng các mô hình state-of-the-art
- Chưa sử dụng pre-trained embeddings
- Một số từ ambiguous vẫn bị gán nhãn sai

### 7.3. Hướng cải thiện

#### 7.3.1. Về mô hình
1. **Bidirectional RNN/LSTM**: Tận dụng context cả 2 chiều
2. **Pre-trained embeddings**: GloVe, FastText, hoặc Contextual embeddings
3. **CRF layer**: Thêm CRF để tận dụng transition probabilities
4. **Attention mechanism**: Cho phép model focus vào context quan trọng

#### 7.3.2. Về dữ liệu
1. **More training data**: Sử dụng thêm dữ liệu từ các nguồn khác
2. **Data augmentation**: Tăng cường dữ liệu bằng các kỹ thuật NLP
3. **Subword tokenization**: Xử lý tốt hơn OOV words

---

## 8. TÀI LIỆU THAM KHẢO

1. Universal Dependencies: https://universaldependencies.org/
2. PyTorch RNN Documentation: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
3. PyTorch LSTM Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
4. Natural Language Processing with Deep Learning (Stanford CS224N)
5. Tài liệu giảng dạy trên lớp


### Files trong project
- [Notebook chính (Lab5_RNN_for_pos_tagging.ipynb)](../../notebook/Lab5/part3/Lab5_RNN_for_pos_tagging.ipynb) - Source code đầy đủ cho bài thực hành
