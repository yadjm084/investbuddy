import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("aapl_news_with_sentiment_scores.csv", parse_dates=["date"])

# Create a histogram of the Sentiment_Score column
fig = px.histogram(
    df, 
    x="Sentiment_Score", 
    nbins=30, 
    title="Histogram of Sentiment Scores",
    labels={"Sentiment_Score": "Sentiment Score", "count": "Frequency"}
)

# Update layout for better readability
fig.update_layout(
    xaxis_title="Sentiment Score",
    yaxis_title="Frequency"
)

fig.show()
