import pandas as pd
import numpy as numpy
import re
from sklearn.model_selection import train_test_split

def clean_roman_urdu(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)

    # Clean Roman Urdu characters and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(file_path):
    print("⏳ Loading dataset...")
    df = pd.read_csv(file_path)

    print(f"📊 Original dataset columns: {list(df.columns)}")
    print(f"📈 Total rows loaded: {len(df):,}")

    text_col = 'Text'
    label_col = 'Sentiment'

    print(f"🔍 Mapping: '{text_col}' -> Text Column | '{label_col}' -> Target Label")

    df = df.dropna(subset=[text_col, label_col])

    print("🧹 Cleaning text records...")
    df['cleaned_text'] = df[text_col].apply(clean_roman_urdu)

    df = df[df['cleaned_text'] != '']

    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    print(f"🔑 Applying structured target mapping: {label_map}")
    df['target_label'] = df[label_col].map(label_map)

    df = df.dropna(subset=['target_label'])
    df['target_label'] = df['target_label'].astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target_label'])

    print(f"✅ Splitting complete! Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    return train_df[['cleaned_text', 'target_label']], test_df[['cleaned_text', 'target_label']]

if __name__ == "__main__":
    # Update this with your exact CSV filename in your folder
    PATH = "dataset/roman_urdu_sentiment_dataset.csv" 
    
    try:
        train, test = prepare_data(PATH)
        
        # Save processed splits to disk
        train.to_csv("train_clean.csv", index=False)
        test.to_csv("test_clean.csv", index=False)
        print("💾 Cleaned datasets written to local disk successfully!")
        
        # Quick sample check
        print("\n👀 Quick check on cleaned sample:")
        print(train.head(2))
        
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        print("\n💡 Hint: Check the file name path string or look inside the CSV file headers.")