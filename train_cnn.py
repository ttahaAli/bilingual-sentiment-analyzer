import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from collections import Counter

# 1. Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("⏳ Loading cleaned data splits...")
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")

train_df['cleaned_text'] = train_df['cleaned_text'].astype(str)
test_df['cleaned_text'] = test_df['cleaned_text'].astype(str)

# 2. Build a Simple Vocabulary mapping for Roman Urdu words
print("🔤 Building token vocabulary index...")
all_words = " ".join(train_df['cleaned_text'].values).split()
word_counts = Counter(all_words)
# Keep words that appear at least twice, reserve 0 for padding, 1 for unknown
vocab = {word: i+2 for i, (word, count) in enumerate(word_counts.items()) if count >= 2}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

VOCAB_SIZE = len(vocab)
MAX_LEN = 32 # Maximum words per review to keep sequence lengths uniform

def text_to_sequence(text):
    tokens = text.split()
    seq = [vocab.get(token, 1) for token in tokens[:MAX_LEN]]
    # Pad with zeros if shorter than MAX_LEN
    return seq + [0] * (MAX_LEN - len(seq))

# Transform datasets to tensor matrices
X_train_seq = np.array([text_to_sequence(t) for t in train_df['cleaned_text']])
X_test_seq = np.array([text_to_sequence(t) for t in test_df['cleaned_text']])

y_train = train_df['target_label'].values
y_test = test_df['target_label'].values

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

# 3. Define the 1D Text CNN Architecture
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim):
        super(TextCNN, self).__init__()
        # Dense representation look-up layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Parallel Convolutional layers tracking phrases of sizes 2, 3, and 4 words
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        # Fully connected final prediction layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len] -> embedding shape: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # Permute to fit Conv1d expectations: [batch_size, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        pooled_outputs = []
        for conv in self.convs:
            # Apply Convolution and activation function
            c = self.relu(conv(embedded))
            # Global Max Pooling over the time dimension
            p = torch.max(c, dim=2)[0]
            pooled_outputs.append(p)
            
        # Concatenate features from all filter sizes
        out = torch.cat(pooled_outputs, dim=1)
        out = self.dropout(out)
        return self.fc(out)

# Hyperparameters
EMBEDDING_DIM = 100
NUM_FILTERS = 64
FILTER_SIZES = [2, 3, 4] # Looking at combinations of bigrams, trigrams, and 4-grams
OUTPUT_DIM = 3

model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM)

# 4. Loss and Optimizer Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
EPOCHS = 5
print(f"\n🎬 Training Text CNN for {EPOCHS} Epochs on CPU...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        
    print(f"Epoch {epoch+1}/{EPOCHS} ➔ Loss: {running_loss / len(train_loader.dataset):.4f}")

# 6. Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"\n✅ Text CNN Test Accuracy: {(correct / total) * 100:.2f}%")

# Save vocabulary state mapping along with weights for future consistency
torch.save({'model_state': model.state_dict(), 'vocab': vocab}, "cnn_baseline.pt")
print("💾 CNN model packet saved safely as 'cnn_baseline.pt'")