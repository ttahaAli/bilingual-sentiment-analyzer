import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("⏳ Loading cleaned data splits...")
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")

# Ensure text rows are parsed as strings properly
train_df['cleaned_text'] = train_df['cleaned_text'].astype(str)
test_df['cleaned_text'] = test_df['cleaned_text'].astype(str)

# Vectorization
print("🧮 Extracting text features via TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000) # Limits vocabulary to top 5000 Roman Urdu words

X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_text']).toarray()
X_test_tfidf = vectorizer.transform(test_df['cleaned_text']).toarray()

y_train = train_df['target_label'].values
y_test = test_df['target_label'].values

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#Create optimized DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the ANN Architecture
class SentimentANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentANN, self).__init__()
        # Layer 1: Input features to hidden representation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Activation function
        self.relu = nn.ReLU()
        # Dropout layer to control overfitting
        self.dropout = nn.Dropout(0.3)
        # Layer 2: Hidden representation to 3 class outputs (Neg, Neu, Pos)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Initialize hyperparameters
INPUT_DIM = X_train_tfidf.shape[1]  # 5000 features
HIDDEN_DIM = 128                    # Nodes in hidden layer
OUTPUT_DIM = 3                      # 3 classes: Negative (0), Neutral (1), Positive (2)

model = SentimentANN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 5. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 6. Training Loop (Running 5 Epochs for quick local verification)
EPOCHS = 5
print(f"\n🚀 Training Baseline ANN for {EPOCHS} Epochs on CPU...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()               # Clear gradients from previous step
        outputs = model(batch_x)            # Forward pass
        loss = criterion(outputs, batch_y)  # Calculate loss
        loss.backward()                     # Backward pass (Backpropagation)
        optimizer.step()                    # Update model weights
        
        running_loss += loss.item() * batch_x.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} ➔ Loss: {epoch_loss:.4f}")

# 7. Model Evaluation
print("\n📊 Evaluating model on the test split...")
model.eval()
correct = 0
total = 0

with torch.no_grad(): # Disable gradient tracking to conserve memory
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = (correct / total) * 100
print(f"✅ Baseline ANN Test Accuracy: {accuracy:.2f}%")

# Save model weights to local disk
torch.save(model.state_dict(), "ann_baseline.pt")
print("💾 Model weights saved safely as 'ann_baseline.pt'")