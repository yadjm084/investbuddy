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
tab_sentiment, tab_forecast, tab_compare, tab_manual = st.tabs([
    "Sentiment Analysis", "Price Forecasting", "Sentiment vs Price", "Manual Text Prediction"
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
            st.write(combined + "…")
            inputs = tokenizer(combined, truncation=True, padding=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            mapped_pred = label_mapping.get(predicted_class, predicted_class)
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
                inputs = tokenizer(
                    user_text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                mapped_pred = label_mapping.get(predicted_class, predicted_class)
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

# ------------------------- Sentiment vs Price Tab -------------------------
with tab_compare:
    st.header("Sentiment vs Price (5-Day Window) with Sentiment Change")
    
    # Load precomputed daily sentiment scores
    news_df = pd.read_csv("aapl_news_with_sentiment_score.csv", parse_dates=["date"])
    # Normalize the date to remove the time component
    news_df["Date"] = news_df["date"].dt.normalize()
    
    # Fetch daily stock prices for 2024
    api_key = 'MGi_WdX9ktIi6maLsK_gcGaa7RrObmQf'
    start, end = "2024-01-01", "2024-12-31"
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    price_data = requests.get(url).json().get("results", [])
    
    # Ensure the field "t" exists. If not, check the API response.
    price_df = pd.DataFrame(price_data)
    if price_df.empty:
        st.error("No price data returned. Please check your API call or date range.")
        st.stop()
    
    price_df["Date"] = pd.to_datetime(price_df["t"], unit="ms")
    price_df.set_index("Date", inplace=True)
    # Normalize the index so that times are removed (e.g., "2024-01-05 05:00:00" becomes "2024-01-05")
    price_df.index = price_df.index.normalize()
    price_df = price_df["c"].rename("Close").to_frame()
    
    # Merge price and sentiment data on date
    sentiment_series = news_df.set_index("Date")["Sentiment_Score"]
    df_compare = price_df.join(sentiment_series, how="inner")
    df_compare.dropna(inplace=True)
    
    st.subheader("Sentiment Score Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=news_df["Sentiment_Score"].dropna(), nbinsx=30))
    fig_hist.update_layout(title="Histogram of Sentiment Scores", 
                           xaxis_title="Score", yaxis_title="Frequency")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Compute 5-day rolling average sentiment and 5-day returns
    df_compare["Sentiment_5d"] = df_compare["Sentiment_Score"].rolling(window=5).mean()
    df_compare["returns_5d"] = df_compare["Close"].pct_change(periods=5)
    
    # Drop rows with NaN values introduced by rolling and pct_change
    df_compare.dropna(subset=["Sentiment_5d", "returns_5d"], inplace=True)
    
    # Standard correlation measures between 5-day rolling sentiment and 5-day returns
    corr_5d = df_compare["returns_5d"].corr(df_compare["Sentiment_5d"], method="pearson")
    spearman_corr_5d = df_compare["returns_5d"].corr(df_compare["Sentiment_5d"], method="spearman")
    kendall_corr_5d = df_compare["returns_5d"].corr(df_compare["Sentiment_5d"], method="kendall")
    st.write(f"**Pearson correlation (5-day returns vs. 5-day sentiment):** {corr_5d:.3f}")
    st.write(f"**Spearman correlation (5-day returns vs. 5-day sentiment):** {spearman_corr_5d:.3f}")
    st.write(f"**Kendall's Tau (5-day returns vs. 5-day sentiment):** {kendall_corr_5d:.3f}")
    
    # Cross-correlation for lags from -5 to +5 (using the 5-day measures)
    def compute_cross_correlation(series1, series2, max_lag=5):
        """Compute cross-correlation for lags in [-max_lag, max_lag]. Positive lag means series2 is shifted forward."""
        lags = range(-max_lag, max_lag + 1)
        correlations = []
        for lag in lags:
            shifted = series2.shift(lag)
            corr = series1.corr(shifted)
            correlations.append((lag, corr))
        return correlations

    cc_values = compute_cross_correlation(df_compare["returns_5d"], df_compare["Sentiment_5d"], max_lag=5)
    cc_df = pd.DataFrame(cc_values, columns=["Lag", "Cross-Correlation"])
    st.subheader("Cross-Correlation for Lags [-5, 5]")
    st.dataframe(cc_df.style.format({"Cross-Correlation": "{:.3f}"}))
    
    # Compute the change in sentiment: difference between consecutive 5-day rolling averages
    df_compare["Sentiment_change_5d"] = df_compare["Sentiment_5d"].diff()
    
    # Correlation between 5-day returns and 5-day sentiment change
    corr_change_5d = df_compare["returns_5d"].corr(df_compare["Sentiment_change_5d"])
    st.write(f"**Pearson correlation (5-day returns vs. 5-day sentiment change):** {corr_change_5d:.3f}")
    
    # Directional P&L over a 5-day window using 5-day average sentiment
    def directional_pnl_5d(close_series, sentiment_series_5d):
        df_5d = pd.DataFrame({
            "future_ret_5d": close_series.pct_change(periods=5).shift(-5),
            "sent_5d": sentiment_series_5d
        }).dropna()
        # Long if the 5-day average sentiment > 0.5, otherwise short
        df_5d["dir"] = np.where(df_5d["sent_5d"] > 0.5, 1, -1)
        df_5d["pnl"] = df_5d["future_ret_5d"] * df_5d["dir"]
        total_pnl = df_5d["pnl"].sum()
        win_rate = (df_5d["pnl"] > 0).mean()
        return total_pnl, win_rate

    total_pnl_5d, win_rate_5d = directional_pnl_5d(df_compare["Close"], df_compare["Sentiment_5d"])
    st.write(f"**Total directional P&L (5-day window based on average sentiment):** {total_pnl_5d:.4f}, **Win rate:** {win_rate_5d:.2%}")
    
    # Directional P&L using sentiment change (if sentiment is increasing, we go long; if decreasing, we go short)
    def directional_pnl_change_5d(close_series, sentiment_change_series):
        df_change = pd.DataFrame({
            "future_ret_5d": close_series.pct_change(periods=5).shift(-5),
            "sent_change": sentiment_change_series
        }).dropna()
        # Long if sentiment change > 0 (i.e., increasing sentiment), short otherwise
        df_change["dir"] = np.where(df_change["sent_change"] > 0, 1, -1)
        df_change["pnl"] = df_change["future_ret_5d"] * df_change["dir"]
        total_pnl = df_change["pnl"].sum()
        win_rate = (df_change["pnl"] > 0).mean()
        return total_pnl, win_rate

    total_pnl_change_5d, win_rate_change_5d = directional_pnl_change_5d(df_compare["Close"], df_compare["Sentiment_change_5d"])
    st.write(f"**Total directional P&L (5-day window based on sentiment change):** {total_pnl_change_5d:.4f}, **Win rate:** {win_rate_change_5d:.2%}")
    
    # Visualize the 5-day rolling sentiment, 5-day returns, and 5-day sentiment change
    st.subheader("5-Day Returns vs. 5-Day Metrics")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_compare.index, y=df_compare["returns_5d"],
                             name="5-Day Returns", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_compare.index, y=df_compare["Sentiment_5d"],
                             name="5-Day Avg Sentiment", line=dict(color='orange'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_compare.index, y=df_compare["Sentiment_change_5d"],
                             name="5-Day Sentiment Change", line=dict(color='green'), yaxis="y3"))
    fig.update_layout(
        title="5-Day Returns, Avg Sentiment, and Sentiment Change",
        yaxis=dict(title="5-Day Returns"),
        yaxis2=dict(title="5-Day Avg Sentiment", overlaying="y", side="right"),
        yaxis3=dict(title="5-Day Sentiment Change", overlaying="y", side="left", anchor="free", position=0.05),
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)