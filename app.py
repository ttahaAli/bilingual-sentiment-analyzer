import sys
# Force python to ignore torchaudio to prevent the WinError 127 DLL crash
sys.modules['torchaudio'] = None

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer


# Set up page styling
st.set_page_config(page_title="Bilingual Sentiment Analyzer", page_icon="🔮", layout="centered")
st.title("🔮 Roman Urdu & English Sentiment Analyzer")
st.markdown("Compare your custom text reviews across **ANN**, **CNN**, and **Transformer** models simultaneously.")

# Label mapping utility
LABEL_MAP = {0: "🔴 Negative", 1: "🟡 Neutral", 2: "🟢 Positive"}

# ==========================================
# 1. ARCHITECTURE DEFINITIONS & CACHING
# ==========================================

class SentimentANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim):
        super().__init__()
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
        pooled_outputs = [torch.max(self.relu(conv(embedded)), dim=2)[0] for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled_outputs, dim=1)))

@st.cache_resource
def load_all_models():
    """Caches weights in memory so the app stays lightning-fast on your CPU."""
    # Load ANN Components
    train_df = pd.read_csv("train_clean.csv")
    ann_vectorizer = TfidfVectorizer(max_features=5000)
    ann_vectorizer.fit(train_df['cleaned_text'].astype(str))
    
    ann_model = SentimentANN(len(ann_vectorizer.get_feature_names_out()), 128, 3)
    ann_model.load_state_dict(torch.load("ann_baseline.pt", map_location=torch.device('cpu'), weights_only = True))
    ann_model.eval()

    # Load CNN Components
    cnn_packet = torch.load("cnn_baseline.pt", map_location=torch.device('cpu'), weights_only=True)
    cnn_vocab = cnn_packet['vocab']
    cnn_model = TextCNN(len(cnn_vocab), 100, 64, [2, 3, 4], 3)
    cnn_model.load_state_dict(cnn_packet['model_state'])
    cnn_model.eval()

    # Load Transformer Components
    # FIX: Point directly to the specialized Roman Urdu model repo
    MODEL_PATH = "Khubaib01/roman-urdu-sentiment-xlm-r" 
    
    trans_tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    trans_model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    trans_model.eval()

    return ann_vectorizer, ann_model, cnn_vocab, cnn_model, trans_tokenizer, trans_model

# Initialize components
try:
    ann_vec, ann_m, cnn_v, cnn_m, trans_tok, trans_m = load_all_models()
    st.success("✅ All 3 models loaded successfully into CPU memory!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("💡 Hint: Ensure 'ann_baseline.pt', 'cnn_baseline.pt', and extracted 'saved_transformer_model' are in this directory.")
    st.stop()

# ==========================================
# 2. RUNNING THE INTERACTIVE INTERFACE
# ==========================================

user_review = st.text_input("✍️ Type your review below (Roman Urdu or English):", 
                           value="delivery thori late thi lekin item bilkul sahi hai")

if st.button("🔮 Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please type a valid phrase.")
    else:
        # Preprocessing function matching Day 1
        clean_text = user_review.lower()
        clean_text = re.sub(r'http\S+|www\S+|<.*?>', '', clean_text)
        clean_text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Layout columns for UI
        col1, col2, col3 = st.columns(3)

        # ---- 1. ANN Inference ----
        ann_input = ann_vec.transform([clean_text]).toarray()
        with torch.no_grad():
            ann_out = ann_m(torch.tensor(ann_input, dtype=torch.float32))
            ann_pred = LABEL_MAP[torch.argmax(ann_out, 1).item()]
        with col1:
            st.metric(label="📊 Baseline ANN", value=ann_pred)
            st.caption("Bag-of-Words Method")

        # ---- 2. CNN Inference ----
        tokens = clean_text.split()
        seq = [cnn_v.get(token, 1) for token in tokens[:32]]
        seq = seq + [0] * (32 - len(seq))
        with torch.no_grad():
            cnn_out = cnn_m(torch.tensor([seq], dtype=torch.long))
            cnn_pred = LABEL_MAP[torch.argmax(cnn_out, 1).item()]
        with col2:
            st.metric(label="🛸 Text CNN", value=cnn_pred)
            st.caption("Sliding Window N-Grams")

        # ---- 3. Transformer Inference ----
        inputs = trans_tok(clean_text, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
        with torch.no_grad():
            trans_out = trans_m(**inputs)
            trans_pred = LABEL_MAP[torch.argmax(trans_out.logits, -1).item()]
        with col3:
            st.metric(label="👑 Transformer (XLM-R)", value=trans_pred)
            st.caption("Contextual Pre-trained Attention")

   # ==========================================
        # 3. VISUAL PERFORMANCE SHOWDOWN
        # ==========================================
        st.markdown("---")
        st.subheader("📊 Model Prediction Probabilities")
        
        # Apply softmax math to transform raw logits into percentage probabilities
        def get_probs(logits_tensor):
            return torch.softmax(logits_tensor, dim=-1).squeeze().tolist()
            
        with torch.no_grad():
            ann_probs = get_probs(ann_out)
            cnn_probs = get_probs(cnn_out)
            trans_probs = get_probs(trans_out.logits)

        # Structure predictions into a clean dataframe chart grid
        chart_data = pd.DataFrame({
            "Sentiment Class": ["Negative", "Neutral", "Positive"],
            "Baseline ANN": ann_probs,
            "Text CNN": cnn_probs,
            "Transformer (XLM-R)": trans_probs
        }).set_index("Sentiment Class")

        # Display interactive chart to the user interface
        st.bar_chart(chart_data)