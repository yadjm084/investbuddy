import pandas as pd
import numpy as np
import re
from langdetect import detect
import joblib
import streamlit as st
from transformers import AutoTokenizer

@st.cache_resource
def load_model():
    try:
        return joblib.load("sentiment_clean.pkl")
    except Exception as e:
        st.warning("Sentiment model unavailable.")
        st.error(f"Load error: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None

@st.cache_resource
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("bert-base-cased")
    except Exception as e:
        st.warning("Tokenizer unavailable.")
        st.error(f"Load error: {e}")
        return None

# Example label mapping (update as needed)
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --- Define a safe text cleaning function ---
def clean_text_safe(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    cleaned_text = text.strip()
    try:
        if detect(cleaned_text) != "en":
            return ""
    except Exception:
        return ""
    return cleaned_text

# --- Load the sentiment model and tokenizer once ---
sentiment_model = load_model()
tokenizer = load_tokenizer()

# --- Process CSV files for years 2021 through 2025 ---
years = [2021, 2022, 2023, 2024, 2025]
all_news = []

for year in years:
    filename = f"aapl_news_{year}.csv"
    try:
        # Expecting each CSV to have columns: 'text' and 'date'
        df = pd.read_csv(filename, parse_dates=["date"])
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping year {year}.")
        continue

    sentiment_scores = []
    print(f"Processing {filename} ...")
    for idx, row in df.iterrows():
        text = row.get("text", "")
        cleaned = clean_text_safe(text)
        if not cleaned:
            sentiment_scores.append(np.nan)
            continue
        try:
            # Use numpy tensors as in your working 2024 code
            tokens = tokenizer(
                cleaned,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="np"
            )
            # Construct a DataFrame to pass to predict() (as in your working example)
            df_in = pd.DataFrame({
                "input_ids": [tokens["input_ids"][0].tolist()],
                "attention_mask": [tokens["attention_mask"][0].tolist()]
            })
            pred = sentiment_model.predict(df_in)[0]
            mapped_pred = label_mapping.get(pred, pred)
        except Exception as e:
            print(f"Error scoring row {idx} in file {filename}: {e}")
            mapped_pred = np.nan
        sentiment_scores.append(mapped_pred)
    
    df["Sentiment_Score"] = sentiment_scores
    df["Year"] = year
    all_news.append(df)

# --- Combine the DataFrames from all years ---
if all_news:
    combined_df = pd.concat(all_news, ignore_index=True)
    combined_df.to_csv("aapl_news_with_sentiment_scores.csv", index=False)
    print("Sentiment scores computed and saved to aapl_news_with_sentiment_score.csv")
else:
    print("No data was processed. Please check your input files.")
