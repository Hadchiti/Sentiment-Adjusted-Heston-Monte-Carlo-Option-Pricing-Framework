# TextBlob-based daily sentiment scoring from news headlines
import os
import pandas as pd
from textblob import TextBlob

from .config import DATA_DIR, TICKER

def polarity(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    try: 
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0
    
def save_sentiment(series: pd.Series, ticker: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}_sentiment.csv")
    series.to_csv(path, header=True)
    print(f"  [saved] {path}")

# Score each article (title & description) with TextBlob polarity then average per calendar day.
# Returns pd.Series - daily polairy in [-1, +1] domain, DatetimeIndex (tz-naive)
def calculate_daily_sentiment(news_df: pd.DataFrame, ticker: str = TICKER) -> pd.Series:
    today = pd.Timestamp.today().normalize()
    empty = pd.Series([0.0], index=pd.DatetimeIndex([today]), name="sentiment")

    if news_df.empty:
        print("  [sentiment] No articles - using neutral sentiment (0.0).")
        save_sentiment(empty, ticker)
        return empty

    df = news_df.copy()
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    df = df.dropna(subset=["publishedAt"])
    if df.empty:
        save_sentiment(empty, ticker)
        return empty
    
    df["text"] = (
        df.get("title", pd.Series(dtype=str)).fillna("") + " " +
        df.get("description", pd.Series(dtype=str)).fillna("")
    ).str.strip()

    print("  [sentiment] Scoring article polarities …")
    df["polarity"] = df["text"].apply(polarity)

    # Strip timezone, normalise to date
    df["date"] = df["publishedAt"].dt.tz_localize(None).dt.normalize()
    daily = df.groupby("date")["polarity"].mean().rename("sentiment")

    # Fill every calendar day in range
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).ffill().bfill()
    daily.name = "sentiment"

    save_sentiment(daily, ticker)
    print(f"  [ok] Sentiment for {len(daily)} days. Mean polarity: {daily.mean():.4f}")
    return daily

# Reindex sentiment onto the price_df trading-day index.
# Forward-fills weekend/holiday gaps.
# Falls back to 0.
def align_sentiment_to_prices(price_df: pd.DataFrame, sentiment: pd.Series) -> pd.Series:
    # Make price dates tz-naive for alignment
    if hasattr(price_df.index, "tz") and price_df.index.tz is not None:
        price_dates = price_df.index.tz_localize(None).normalize()
    else:
        price_dates = price_df.index.normalize()

    aligned = sentiment.reindex(price_dates, method="ffill").fillna(0.0)
    aligned.index = price_df.index
    return aligned
 