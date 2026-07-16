# 🔮 Bilingual Roman Urdu & English Sentiment Analyzer

An end-to-end Deep Learning workspace that compares three generations of NLP architectures (ANN, 1D-CNN, and Transformer) classifying customer reviews.

## 🚀 Live Demo
👉 **[Experience the Live Application Here](https://bilingual-sentiment-analyzer-gaaiuhbvd3a2rsn5ec6ziu.streamlit.app)**

This interactive web dashboard allows users to test the pipeline live by inputting custom customer reviews in Roman Urdu or English. The cloud application processes raw text input on the fly and runs simultaneous inference across the deployed model architectures. By visualizing live prediction distributions side by side, the application provides an interactive benchmark demonstrating how different deep learning approaches extract linguistic patterns and handle code-switched or mixed-language datasets.


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

### 🧠 Key Engineering Takeaways
- **Resource Caching**: Implemented Streamlit `@st.cache_resource` to isolate large deep learning network weights in memory, enabling real-time CPU evaluation.
- **Dependency Isolation**: Isolated local CPU builds from heavy cloud GPU configurations, building a flexible cross-platform environment setup.
- **Multilingual Tokenization**: Configured an explicit `XLMRobertaTokenizer` architecture to handle non-standard subword patterns in Roman Urdu text.

## 💻 How to Run Locally
1. Clone the repo: `git clone https://github.com/ttahaAli/bilingual-sentiment-analyzer`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python -m streamlit run app.py`
