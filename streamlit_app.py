import streamlit as st
import requests
import datetime
import re
import pandas as pd
import joblib
import numpy as np
import torch
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import plotly.graph_objects as go
from math import sqrt
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------- Fonction de nettoyage -------------------------
def clean_text(text):
    """Convert text to lowercase, remove URLs, mentions, hashtags, punctuation, and extra spaces.
    Only returns English text."""
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

# ------------------------- Tabs ----------------------------------
tab_sentiment, tab_forecast, tab_recommendation = st.tabs([
    "Sentiment Analysis", "Price Forecasting", "Recommendation"
])


# ------------------------- Sentiment Tab -------------------------
with tab_sentiment:
    st.header("Sentiment Analysis")
    st.write(f"Fetching Reddit & Polygon news for **{stock_symbol}**")

    @st.cache_resource
    def load_model():
        try:
            # Load a fine-tuned sentiment model directly from Hugging Face
            return AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        except Exception as e:
            st.warning("Sentiment model unavailable.")
            st.error(f"Load error: {e}")
            import traceback
            st.text(traceback.format_exc())
            return None

    @st.cache_resource
    def load_tokenizer():
        try:
            return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        except Exception as e:
            st.warning("Tokenizer unavailable.")
            return None

    model = load_model()
    tokenizer = load_tokenizer()

    if model is None or tokenizer is None:
        st.warning("Sentiment can't be analyzed for now.")
    else:
        # Fetch texts from two sources
        texts = []
        for func in (
            lambda t: requests.get(f"https://reddit-scraper2.p.rapidapi.com/search_posts_v3?query={t}&sort=RELEVANCE&time=day").json().get("data", []),
            lambda t: requests.get(f"https://api.polygon.io/v2/reference/news?ticker={t}&limit=10&apiKey=MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf").json().get("results", [])
        ):
            try:
                for item in func(stock_symbol):
                    txt = item.get("content", item.get("description", ""))
                    cleaned = clean_text(txt)
                    if cleaned:
                        texts.append(cleaned)
            except Exception as e:
                st.error(f"Fetch error: {e}")

        combined = " ".join(texts)
        if combined:
            inputs = tokenizer(combined, truncation=True, padding=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            mapped_pred = label_mapping.get(predicted_class, predicted_class)
            st.success(f"Sentiment: **{mapped_pred}**")
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
    df_stock = pd.DataFrame(requests.get(url).json().get('results', []))
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
        # Build a dictionary of Series then concat once for efficiency
        data = {}
        for col in ['Open','High','Low','Close','Volume']:
            data[col] = df[col]
            for lag in range(1, n_lags + 1):
                data[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Concatenate all at once and drop NaNs
        return pd.concat(data, axis=1).dropna()

    train_lagged = create_lag_df(train_df)
    val_lagged = create_lag_df(pd.concat([train_df.tail(24), val_df]))

    X_train, y_train = train_lagged.drop('Close', axis=1), train_lagged['Close']
    X_val, y_val = val_lagged.drop('Close', axis=1), val_lagged['Close']

    # Hyperparameter tuning with TimeSeriesSplit
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                        param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    st.write("Best XGB params:", grid.best_params_)

    # Validation performance
    val_pred = best_model.predict(X_val)
    st.write(f"Validation RMSE: {sqrt(mean_squared_error(y_val, val_pred)):.3f}")
    st.write(f"Validation MAE: {mean_squared_error(y_val, val_pred, squared=False):.3f}")
    st.write(f"Validation MAPE: {np.mean(np.abs((y_val - val_pred) / y_val)) * 100:.2f}%")

    # Retrain on full data
    full_lagged = create_lag_df(df_stock)
    best_model.fit(full_lagged.drop('Close', axis=1), full_lagged['Close'])

    # Rolling forecast for next 240 hours
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

    # Plot last 6 months + forecast
    one_month_ago = df_stock.index.max() - pd.DateOffset(months=6)
    df_recent = df_stock[df_stock.index >= one_month_ago]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Close'], name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_index, y=xgb_series, name='XGBoost Forecast'))
    fig.update_layout(title=f"{stock_symbol.upper()} Price Forecast (Next 240h)",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Recommendation Tab -------------------------
with tab_recommendation:
    st.header("Buy / Sell / Hold Recommendation")
    st.write(f"Generating recommendation for **{stock_symbol.upper()}**")

    # Step 1: Sentiment Analysis
    try:
        texts = []
        for func in (
            lambda t: requests.get(
                f"https://reddit-scraper2.p.rapidapi.com/search_posts_v3?query={t}&sort=RELEVANCE&time=day"
            ).json().get("data", []),
            lambda t: requests.get(
                f"https://api.polygon.io/v2/reference/news?ticker={t}&limit=10&apiKey=MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf"
            ).json().get("results", [])
        ):
            for item in func(stock_symbol):
                txt = item.get("content", item.get("description", ""))
                cleaned = clean_text(txt)
                if cleaned:
                    texts.append(cleaned)

        combined_text = " ".join(texts)

        if combined_text:
            sentiment_inputs = tokenizer(
                combined_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            sentiment_outputs = model(**sentiment_inputs)
            sentiment_logits = sentiment_outputs.logits
            sentiment_prediction = torch.argmax(sentiment_logits, dim=1).item()
            sentiment_score = label_mapping.get(sentiment_prediction, 0)
            st.write(f"📝 Latest Sentiment Score: **{sentiment_score}**")
        else:
            st.warning("No recent sentiment data found.")
            sentiment_score = 0
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        sentiment_score = 0

    # Step 2: Price Forecasting using all predicted points
    try:
        latest_price = df_stock['Close'].iloc[-1]

        # Récupération de toutes les valeurs prédites par XGBoost (les 240 points)
        all_predicted_prices = xgb_series  # xgb_series est déjà calculé dans l'onglet Price Forecasting
        average_predicted_price = all_predicted_prices.mean()
        
        # On peut choisir de baser la "tendance" sur la différence entre la moyenne prévue et le prix actuel
        price_change = average_predicted_price - latest_price

        st.write(f"💰 Latest Price: **{latest_price:.2f}**")
        st.write("🔮 All XGBoost Forecast Points:")
        st.write(all_predicted_prices)

        st.write(f"💹 Average Predicted Price (over all forecast points): **{average_predicted_price:.2f}**")
        st.write(f"📈 Expected Change (vs. latest price): **{price_change:.2f}**")
    except Exception as e:
        st.error(f"Error fetching price forecast: {e}")
        price_change = 0

    # Step 3: Decision Logic
    if sentiment_score > 0 and price_change > 0:
        recommendation = "BUY ✅"
        explanation = "Positive sentiment and overall price increase predicted."
    elif sentiment_score < 0 and price_change < 0:
        recommendation = "SELL 🛑"
        explanation = "Negative sentiment and overall price decrease predicted."
    else:
        recommendation = "HOLD 🤔"
        explanation = "Mixed signals, better to hold for now."

    st.subheader(f"Recommendation: {recommendation}")
    st.info(explanation)