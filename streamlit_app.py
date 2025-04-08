import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Limiter le parallélisme pour éviter certains problèmes

import streamlit as st
import requests
import datetime
import re
import pandas as pd
import joblib
import numpy as np
import torch
from langdetect import detect
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import plotly.graph_objects as go
from math import sqrt
import warnings
import time
from collections import Counter

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------- Helper: API Fetch with Retries -------------------------
def fetch_with_retries(url, headers=None, retries=3, delay=2, timeout=10):
    """
    Helper function to fetch data from a URL with retries.
    Logs errors using st.error and returns the response if successful,
    or None after all retries fail.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 502:
                st.error(f"Received a 502 Bad Gateway error for URL {url}")
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            st.error(f"Attempt {attempt+1} failed for URL {url}: {e}")
            time.sleep(delay)
    return None

# ------------------------- Fonction de nettoyage -------------------------
def clean_text(text):
    """
    Convertit le texte en minuscules, supprime URLs, mentions, hashtags, ponctuation et espaces superflus.
    Retourne uniquement le texte en anglais.
    """
    # Si text n'est pas une chaîne, retourner une chaîne vide
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)      # Remove mentions
    text = re.sub(r"#\w+", "", text)       # Remove hashtags
    text = re.sub(r"\s+", " ", text)       # Remove extra spaces
    cleaned_text = text.strip()
    try:
        if detect(cleaned_text) != "en":
            return ""
    except Exception:
        return ""
    return cleaned_text

# ------------------------- Configuration -------------------------
st.set_page_config(page_title="Stock Sentiment & Price Forecast App", layout="centered")
query_params = st.query_params
stock_symbol = query_params.get("stock", "AAPL")  # Default to AAPL

# Define sentiment label mapping: 0 -> -1, 1 -> 0, 2 -> 1
label_mapping = {0: -1, 1: 0, 2: 1}

# RapidAPI headers for Reddit
reddit_headers = {
    "x-rapidapi-key": "39408f7417msh190420cfe381944p16bd39jsndb389cd3e14e",
    "x-rapidapi-host": "reddit-scraper2.p.rapidapi.com"
}

# ------------------------- Tabs ----------------------------------
tab_sentiment, tab_forecast, tab_recommendation = st.tabs([
    "Sentiment Analysis", "Price Forecasting", "Recommendation"
])


# ------------------------- Sentiment Tab -------------------------
with tab_sentiment:
    st.header("Sentiment Analysis")
    st.write(f"Fetching Reddit & Polygon news for **{stock_symbol}**")

    @st.cache_resource
    def load_local_model():
        try:
            model = joblib.load("sentiment_classifier.pkl")
            return model
        except Exception as e:
            st.warning("Sentiment model unavailable.")
            st.error(f"Load error: {e}")
            import traceback
            st.text(traceback.format_exc())
            return None

    model = load_local_model()

    if model is None:
        st.warning("Sentiment can't be analyzed for now.")
    else:
        texts = []
        # Traitement des données Reddit avec vérification de type
        reddit_url = f"https://reddit-scraper2.p.rapidapi.com/search_posts_v3?query={stock_symbol}&sort=RELEVANCE&time=day"
        reddit_response = fetch_with_retries(reddit_url, headers=reddit_headers)
        if reddit_response:
            try:
                reddit_items = reddit_response.json().get("data", [])
                for item in reddit_items:
                    # Récupère le texte à partir de 'content' ou 'description'
                    txt = item.get("content") or item.get("description", "")
                    if not isinstance(txt, str):
                        continue  # Passe à l'élément suivant si txt n'est pas une chaîne
                    cleaned = clean_text(txt)
                    if cleaned:
                        texts.append(cleaned)
            except Exception as e:
                st.error(f"Error processing Reddit data: {e}")
        else:
            st.error("Failed to fetch Reddit data after multiple attempts.")

        # Traitement des données de Polygon
        polygon_url = f"https://api.polygon.io/v2/reference/news?ticker={stock_symbol}&limit=10&apiKey=MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf"
        polygon_response = fetch_with_retries(polygon_url)
        if polygon_response:
            try:
                polygon_items = polygon_response.json().get("results", [])
                for item in polygon_items:
                    txt = item.get("content") or item.get("description", "")
                    if not isinstance(txt, str):
                        continue
                    cleaned = clean_text(txt)
                    if cleaned:
                        texts.append(cleaned)
            except Exception as e:
                st.error(f"Error processing Polygon data: {e}")
        else:
            st.error("Failed to fetch Polygon news data after multiple attempts.")

        if texts:
            try:
                # Fusionner tous les textes en une seule chaîne
                combined_text = " ".join(texts)
                # Créer un DataFrame avec une seule ligne contenant le texte combiné
                input_df = pd.DataFrame({'text': [combined_text]})
                # Faire une seule prédiction
                prediction = model.predict(input_df)[0]
                mapped_pred = label_mapping.get(prediction, prediction)
                st.success(f"Sentiment unique: **{mapped_pred}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                st.text(traceback.format_exc())
        else:
            st.info("No text data available.")


# ------------------------- Price Forecasting Tab -------------------------
with tab_forecast:
    st.header("Price Forecasting")
    st.write(f"Forecasting next 240 hours for **{stock_symbol.upper()}** using Rolling XGBoost with CV.")
    st.info("Fetching stock data…")
    
    api_key = 'MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf'
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    url = (
        f'https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/day/'
        f'{from_date}/{to_date}?adjusted=true&sort=asc&limit=-1&apiKey={api_key}'
    )
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        st.stop()
    data = response.json()
    if 'results' not in data or not data['results']:
        st.error("No data available in the response.")
        st.stop()

    df_stock = pd.DataFrame(data.get('results', []))
    if df_stock.empty:
        st.error("No data fetched. Please check the stock symbol and API key.")
        st.stop()

    df_stock['Date'] = pd.to_datetime(df_stock['t'], unit='ms')
    df_stock.set_index('Date', inplace=True)
    df_stock.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}, inplace=True)
    df_stock = df_stock.asfreq('h', method='ffill')

    st.write("DataFrame shape:", df_stock.shape)
    st.subheader("Last Fetched Data Point")
    st.write(f"- Date: {df_stock.index[-1]}")
    st.write(f"- Close: {df_stock['Close'].iloc[-1]:.2f}")

    # Train/Validation split (80/20)
    split = int(len(df_stock) * 0.8)
    train_df, val_df = df_stock.iloc[:split], df_stock.iloc[split:]

    def create_lag_df(df, n_lags=24):
        data = {}
        for col in ['Open','High','Low','Close','Volume']:
            data[col] = df[col]
            for lag in range(1, n_lags + 1):
                data[f"{col}_lag{lag}"] = df[col].shift(lag)
        return pd.concat(data, axis=1).dropna()

    train_lagged = create_lag_df(train_df)
    val_lagged = create_lag_df(pd.concat([train_df.tail(24), val_df]))

    X_train, y_train = train_lagged.drop('Close', axis=1), train_lagged['Close']
    X_val, y_val = val_lagged.drop('Close', axis=1), val_lagged['Close']

    # Adjust TimeSeriesSplit based on data size
    min_samples = 6  
    if len(X_train) < min_samples:
        n_splits = 2
    else:
        n_splits = 5

    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    grid = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid, 
        cv=tscv, 
        scoring='neg_root_mean_squared_error',
        n_jobs=1  # Utilise 1 cpu pour éviter les problèmes de parallélisme
    )
    
    try:
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        st.write("Best XGB params:", grid.best_params_)

        val_pred = best_model.predict(X_val)
        st.write(f"Validation RMSE: {sqrt(mean_squared_error(y_val, val_pred)):.3f}")
        st.write(f"Validation MAE: {mean_absolute_error(y_val, val_pred):.3f}")
        st.write(f"Validation MAPE: {np.mean(np.abs((y_val - val_pred) / y_val)) * 100:.2f}%")

        full_lagged = create_lag_df(df_stock)
        best_model.fit(full_lagged.drop('Close', axis=1), full_lagged['Close'])

        history = df_stock.copy()
        preds = []
        for _ in range(240):
            last_features = create_lag_df(history).iloc[[-1]].drop('Close', axis=1)
            pred = best_model.predict(last_features)[0]
            preds.append(pred)
            next_time = history.index[-1] + pd.DateOffset(hours=1)
            new_row = history.iloc[-1].copy()
            new_row['Close'] = pred
            new_row.name = next_time
            history = pd.concat([history, new_row.to_frame().T])

        forecast_index = pd.date_range(start=df_stock.index[-1] + pd.DateOffset(hours=1), periods=240, freq='H')
        xgb_series = pd.Series(preds, index=forecast_index)

        one_month_ago = df_stock.index.max() - pd.DateOffset(months=6)
        df_recent = df_stock[df_stock.index >= one_month_ago]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Close'], name='Historical'))
        fig.add_trace(go.Scatter(x=forecast_index, y=xgb_series, name='XGBoost Forecast'))
        fig.update_layout(title=f"{stock_symbol.upper()} Price Forecast (Next 240h)",
                          xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        
    except ValueError as e:
        st.error(f"Error in model training: {e}")
        st.warning("Reducing number of splits due to limited data...")
        n_splits = min(3, len(X_train) - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        grid = GridSearchCV(
            XGBRegressor(objective='reg:squarederror', random_state=42),
            param_grid, 
            cv=tscv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        # Continue avec le reste du processus...

# ------------------------- Recommendation Tab -------------------------
with tab_recommendation:
    st.header("Buy / Sell / Hold Recommendation")
    st.write(f"Generating recommendation for **{stock_symbol.upper()}**")

    try:
        texts = []
        for func in (
            lambda t: requests.get(f"https://reddit-scraper2.p.rapidapi.com/search_posts_v3?query={t}&sort=RELEVANCE&time=day").json().get("data", []),
            lambda t: requests.get(f"https://api.polygon.io/v2/reference/news?ticker={t}&limit=10&apiKey=MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf").json().get("results", [])
        ):
            try:
                for item in func(stock_symbol):
                    txt = item.get("content") or item.get("description", "")
                    if not isinstance(txt, str):
                        continue
                    cleaned = clean_text(txt)
                    if cleaned:
                        texts.append(cleaned)
            except Exception as e:
                st.error(f"Fetch error in Recommendation tab: {e}")

        if texts and model is not None:
            try:
                input_df = pd.DataFrame({'text': texts})
                predictions = model.predict(input_df)
                sentiment_score = sum(label_mapping.get(p, 0) for p in predictions) / len(predictions)
                st.write(f"📝 Average Sentiment Score: **{sentiment_score:.2f}**")
            except Exception as e:
                st.error(f"Sentiment prediction failed: {e}")
                sentiment_score = 0
        else:
            st.warning("No recent sentiment data found.")
            sentiment_score = 0
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        sentiment_score = 0

    try:
        latest_price = df_stock['Close'].iloc[-1]
        average_predicted_price = xgb_series.mean()
        price_change = average_predicted_price - latest_price

        st.write(f"💰 Latest Price: **{latest_price:.2f}**")
        st.write(f"💹 Average Predicted Price (over all forecast points): **{average_predicted_price:.2f}**")
        st.write(f"📈 Expected Change (vs. latest price): **{price_change:.2f}**")
    except Exception as e:
        st.error(f"Error fetching price forecast: {e}")
        price_change = 0

    if sentiment_score > 0.5 and price_change > 0:
        recommendation = "STRONG BUY ✅✅"
        explanation = "Very positive sentiment and strong price increase predicted."
    elif sentiment_score > 0 and price_change > 0:
        recommendation = "BUY ✅"
        explanation = "Positive sentiment and price increase predicted."
    elif sentiment_score < -0.5 and price_change < 0:
        recommendation = "STRONG SELL 🛑🛑"
        explanation = "Very negative sentiment and strong price decrease predicted."
    elif sentiment_score < 0 and price_change < 0:
        recommendation = "SELL 🛑"
        explanation = "Negative sentiment and price decrease predicted."
    else:
        recommendation = "HOLD 🤔"
        explanation = "Mixed signals or neutral outlook."

    st.subheader(f"Recommendation: {recommendation}")
    st.info(explanation)
