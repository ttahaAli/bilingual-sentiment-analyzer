# 🔮 Bilingual Roman Urdu & English Sentiment Analyzer

An end-to-end Deep Learning workspace that compares three generations of NLP architectures (ANN, 1D-CNN, and Transformer) classifying customer reviews.

## 🚀 Live Demo
[Insert Streamlit Cloud Link Here Later]

## 🛠️ Project Architecture & Comparison
This project evaluates text simultaneously across three distinct methodologies:
1. **Baseline ANN**: Uses TF-IDF (Bag-of-Words) features fed into dense layers. Fast, but struggles with context shifts.
2. **Text 1D-CNN**: Utilizes word embeddings and multi-kernel sliding windows (Bigrams/Trigrams) to capture localized phrases.
3. **Transformer (XLM-RoBERTa)**: Leverages a state-of-the-art multilingual model trained on real-world Roman Urdu marketplace data.

## 📊 Real-World Evaluation & Edge Cases
During testing, a fascinating linguistic edge case was discovered using the phrase: 
> *"delivery thori late thi lekin item bilkul sahi hai"*

- **ANN Classifer** ➔ Predicted `Negative` (Tricked by the standalone keyword "late").
- **CNN Classifier** ➔ Predicted `Neutral` (Balanced out the positive and negative phrase structures).
- **Transformer** ➔ Predicted `Negative` (Reflects real-world e-commerce data trends where logistics failures heavily weigh down customer sentiment).

## 💻 How to Run Locally
1. Clone the repo: `git clone https://github.com/ttahaAli/bilingual-sentiment-analyzer`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python -m streamlit run app.py`
