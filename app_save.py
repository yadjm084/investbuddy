import streamlit as st
import requests
import datetime
import re
import pandas as pd
import joblib
import numpy as np
from langdetect import detect
from transformers import AutoTokenizer
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
tab_sentiment, tab_forecast, tab_compare, tab_manual = st.tabs(["Sentiment Analysis", "Price Forecasting", "Sentiment vs Price", "Manual Text Prediction"])

# ------------------------- Sentiment Tab -------------------------
with tab_sentiment:
    st.header("Sentiment Analysis")
    st.write(f"Fetching Reddit & Polygon news for **{stock_symbol}**")

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
        except:
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
                    txt = item.get("content", item.get("description",""))
                    cleaned = clean_text(txt)
                    if cleaned:
                        texts.append(cleaned)
            except Exception as e:
                st.error(f"Fetch error: {e}")

        combined = " ".join(texts)
        if combined:
            st.write(combined + "…")
            tokens = tokenizer(combined, truncation=True, padding=True, return_tensors="np")
            df_in = pd.DataFrame({
                "input_ids": [tokens["input_ids"][0].tolist()],
                "attention_mask": [tokens["attention_mask"][0].tolist()]
            })
            pred = model.predict(df_in)[0]
            mapped_pred = label_mapping.get(pred, pred)
            st.success(f"Sentiment: **{mapped_pred}**")
        else:
            st.info("No text data available.")

# ------------------------- Manual Text Prediction Tab -------------------------
with tab_manual:
    st.header("Manual Text Sentiment Prediction")

    model = load_model()
    tokenizer = load_tokenizer()

    user_text = st.text_area("Paste a news headline or article snippet below:")

    # Submit button to trigger prediction
    submit = st.button("Submit")

    if submit:
        if user_text and model is not None and tokenizer is not None:
            try:
                tokens = tokenizer(
                    user_text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="np"
                )
                df_in = pd.DataFrame({
                    "input_ids": [tokens["input_ids"][0].tolist()],
                    "attention_mask": [tokens["attention_mask"][0].tolist()]
                })
                pred = model.predict(df_in)[0]
                mapped_pred = label_mapping.get(pred, pred)
                st.success(f"Predicted Sentiment: **{mapped_pred}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        elif user_text:
            st.warning("Model or tokenizer is not loaded correctly.")

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
    split = int(len(df_stock)*0.8)
    train_df, val_df = df_stock.iloc[:split], df_stock.iloc[split:]

    def create_lag_df(df, n_lags=24):
        # Build a dictionary of Series then concat once for efficiency
        data = {}
        for col in ['Open','High','Low','Close','Volume']:
            data[col] = df[col]
            for lag in range(1, n_lags+1):
                data[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Concatenate all at once and drop NaNs
        return pd.concat(data, axis=1).dropna()

    train_lagged = create_lag_df(train_df)
    val_lagged = create_lag_df(pd.concat([train_df.tail(24), val_df]))

    X_train, y_train = train_lagged.drop('Close', axis=1), train_lagged['Close']
    X_val, y_val     = val_lagged.drop('Close', axis=1), val_lagged['Close']

    # Hyperparameter tuning with TimeSeriesSplit
    param_grid = {'n_estimators': [100,200], 'max_depth': [3,5]}
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
    st.write(f"Validation MAPE: {np.mean(np.abs((y_val - val_pred)/y_val))*100:.2f}%")

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

    forecast_index = pd.date_range(start=df_stock.index[-1]+pd.DateOffset(hours=1), periods=240, freq='H')
    xgb_series = pd.Series(preds, index=forecast_index)

    # Plot last 6 months + forecast
    one_month_ago = df_stock.index.max() - pd.DateOffset(months=6)
    df_recent = df_stock[df_stock.index>=one_month_ago]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Close'], name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_index, y=xgb_series, name='XGBoost Forecast'))
    fig.update_layout(title=f"{stock_symbol.upper()} Price Forecast (Next 240h)", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Sentiment vs Price Tab -------------------------
with tab_compare:
    st.header("Sentiment vs Price (2024)")
    news_df = pd.read_csv("aapl_news_2024.csv", parse_dates=["date"])
    news_df["Date"] = news_df["date"].dt.normalize()

    api_key = 'MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf'
    start, end = "2024-01-01", "2024-12-31"
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    price_df = pd.DataFrame(requests.get(url).json().get("results", []))
    price_df["Date"] = pd.to_datetime(price_df["t"], unit="ms")
    price_df.set_index("Date", inplace=True)
    price_df = price_df["c"].rename("Close").to_frame()

    sentiment_model = load_model()
    tokenizer = load_tokenizer()

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

    sentiment_scores = []
    st.info("Running sentiment analysis row by row using predict()…")
    for idx, row in news_df.iterrows():
        text = row.get("text", "")
        cleaned = clean_text_safe(text)
        if not cleaned:
            sentiment_scores.append(np.nan)
            continue
        try:
            tokens = tokenizer(cleaned, truncation=True, padding=True, max_length=512, return_tensors="np")
            df_in = pd.DataFrame({
                "input_ids": [tokens["input_ids"][0].tolist()],
                "attention_mask": [tokens["attention_mask"][0].tolist()]
            })
            pred = sentiment_model.predict(df_in)[0]
            mapped_pred = label_mapping.get(pred, pred)
        except Exception as e:
            print(f"Error scoring row {idx}: {e}")
            mapped_pred = np.nan
        sentiment_scores.append(mapped_pred)

    news_df["Sentiment_Score"] = sentiment_scores
    news_df.to_csv("aapl_news_with_sentiment_score.csv", index=False)

    daily_sentiment = news_df.groupby("Date")["Sentiment_Score"].mean()
    df_compare = price_df.join(daily_sentiment).dropna()

    st.subheader("Sentiment Score Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=news_df["Sentiment_Score"].dropna(), nbinsx=30))
    fig_hist.update_layout(title="Histogram of Sentiment Scores", xaxis_title="Score", yaxis_title="Frequency")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_compare.index, y=df_compare["Close"], name="Close Price"))
    fig.add_trace(go.Scatter(x=df_compare.index, y=df_compare["Sentiment_Score"], name="Avg Sentiment Score", yaxis="y2"))
    fig.update_layout(title="AAPL 2024: Price vs Sentiment Score", yaxis_title="Close Price", 
                      yaxis2=dict(title="Sentiment Score", overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)

    corr = df_compare["Close"].pct_change().corr(df_compare["Sentiment_Score"])
    st.write(f"Pearson correlation (returns vs sentiment): {corr:.3f}")

    def directional_pnl(actual, sentiment):
        df = pd.DataFrame({"ret": actual.pct_change().shift(-1), "sent": sentiment})
        df["dir"] = np.where(df["sent"] > 0.5, 1, -1)
        df["pnl"] = df["ret"] * df["dir"]
        return df["pnl"].sum(), (df["pnl"] > 0).mean()

    total_pnl, win_rate = directional_pnl(df_compare["Close"], df_compare["Sentiment_Score"])
    st.write(f"Total directional P&L: {total_pnl:.4f}, Win rate: {win_rate:.2%}")
