import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# ==========================================
# 1. SETUP LOGIC FOR BOTH MODEL DESIGNS
# ==========================================

# ANN Setup Setup
train_df = pd.read_csv("train_clean.csv")
train_df['cleaned_text'] = train_df['cleaned_text'].astype(str)
ann_vectorizer = TfidfVectorizer(max_features=5000)
ann_vectorizer.fit(train_df['cleaned_text'])

class SentimentANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

ann_model = SentimentANN(len(ann_vectorizer.get_feature_names_out()), 128, 3)
ann_model.load_state_dict(torch.load("ann_baseline.pt"))
ann_model.eval()

# CNN Setup Setup
cnn_packet = torch.load("cnn_baseline.pt")
vocab = cnn_packet['vocab']
VOCAB_SIZE = len(vocab)
MAX_LEN = 32

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        pooled_outputs = []
        for conv in self.convs:
            c = self.relu(conv(embedded))
            p = torch.max(c, dim=2)[0]
            pooled_outputs.append(p)
        return self.fc(self.dropout(torch.cat(pooled_outputs, dim=1)))

cnn_model = TextCNN(VOCAB_SIZE, 100, 64, [2,3,4], 3)
cnn_model.load_state_dict(cnn_packet['model_state'])
cnn_model.eval()

def text_to_sequence(text):
    tokens = text.lower().split()
    seq = [vocab.get(token, 1) for token in tokens[:MAX_LEN]]
    return seq + [0] * (MAX_LEN - len(seq))

# ==========================================
# 2. EVALUATION PIPELINE
# ==========================================

complex_review = "delivery thori late thi lekin item bilkul sahi hai"
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"📝 Evaluation Review: \"{complex_review}\"\n")

# Process & Run ANN Prediction
ann_vec = ann_vectorizer.transform([complex_review]).toarray()
with torch.no_grad():
    ann_out = ann_model(torch.tensor(ann_vec, dtype=torch.float32))
    ann_pred = label_mapping[torch.argmax(ann_out, 1).item()]
print(f"🤖 ANN Prediction ('Bag-of-Words'): {ann_pred}")

# Process & Run CNN Prediction
cnn_seq = text_to_sequence(complex_review)
with torch.no_grad():
    cnn_out = cnn_model(torch.tensor([cnn_seq], dtype=torch.long))
    cnn_pred = label_mapping[torch.argmax(cnn_out, 1).item()]
print(f"🛸 CNN Prediction ('N-Gram Sequence'): {cnn_pred}")
