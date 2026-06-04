import os
import time
import datetime
import requests
import pandas as pd
import yfinance as yf

from .config import (
    TICKER, NEWS_API_KEY, HISTORICAL_PERIOD,
    NEWS_LOOKBACK_DAYS, DATA_DIR,
)

# Helper to check the existance of a directory
def ensure_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

# Helper to save a file
def save_file(df: pd.DataFrame, filename: str) -> None:
    ensure_dir()
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    print(f"  [saved] {path}")

######################################
## Stock Prices
######################################

# Get 1-year daily OHLCV data from Yahoo Finance
def get_stock_prices(ticker: str = TICKER,
                     period: str = HISTORICAL_PERIOD) -> pd.DataFrame:
    print(f"[data] Fetching {period} price history for {ticker} ...")
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                raise ValueError("Empty response from yfinance")
            save_file(df, f"{ticker}_prices.csv")
            print(f"  [ok] {len(df)} trading days.")
            return df
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [warn] attempt {attempt+1}: {exc}. Retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(f"Cannot fetch price data for {ticker}.")


######################################
## Option Chain
######################################

# Get calls and puts for the nearest future expiry from Yahoo Finance
# Returns: calls_df, puts_df and expiry_date
def get_option_chain(ticker: str = TICKER) -> tuple[pd.DataFrame, pd.DataFrame, datetime.date]:
    print(f"[data] Fetching option chain for {ticker} ...")
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
            if not expirations:
                raise ValueError("No options listed")
            
            # Get all option expiration dates for this stock that are after today
            # If none, throw an error.
            today = datetime.date.today()
            future = [e for e in expirations
                        if datetime.date.fromisoformat(e) >= today + datetime.timedelta(days=21)]
            if not future:
                raise ValueError("All listed expiries are today or past")
            
            # Take the earliest expiry date
            future_sorted = sorted(future)
            nearest = future_sorted[0]
            expiry_date = datetime.date.fromisoformat(nearest)
            chain = tk.option_chain(nearest)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            for df in (calls, puts):
                df["ticker"] = ticker
                df["expiry"] = nearest
            
            save_file(calls, f"{ticker}_calls_{nearest}.csv")
            save_file(puts, f"{ticker}_puts_{nearest}.csv")
            print(f"  [ok] expiry={nearest} | calls = {len(calls)} | puts = {len(puts)}")
            return calls, puts, expiry_date
        
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [warn] attempt {attempt+1}: {exc}. Retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(f"Cannot fetch option chain for {ticker}.")

######################################
## News
######################################

# Pull recent headlines from NewsAPI. 
# Falls back to empty DataFrame (neutral Sentiment) if the key is missing or the call fails.
def get_news(ticker: str = TICKER, api_key: str = NEWS_API_KEY, lookback_days: int = NEWS_LOOKBACK_DAYS) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["publishedAt", "source", "title", "description"])

    if not api_key or api_key == "YOUR_NEWSAPI_KEY_HERE":
        print("  [warn] NEWS_API_KEY not set - using neutral sentiment (polarity = 0).")
        save_file(empty, f"{ticker}_news.csv")
        return empty

    print(f"[data] Fetching news for '{ticker}' (last {lookback_days} days) ...")
    from_date = (
        datetime.date.today() - datetime.timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    params = {
        "q":        ticker,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 100,
        "apiKey":   api_key,
    }

    for attempt in range(3):
        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "ok":
                raise ValueError(f"NewsAPI: {data.get('message', 'unknown error')}")
            
            articles = data.get("articles", [])
            if not articles:
                print("  [warn] NewsAPI returned 0 articles.")
                save_file(empty, f"{ticker}_news.csv")
                return empty
            
            df = pd.DataFrame([{
                "publishedAt": a.get("publishedAt", ""),
                "source":      a.get("source", {}).get("name", ""),
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
            } for a in articles])
            df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
            save_file(df, f"{ticker}_news.csv")
            print(f"  [ok] {len(df)} articles.")
            return df
        
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [warn] attempt {attempt+1}: {exc}. Retrying in {wait}s ...")
            time.sleep(wait)
    print("  [error] All NewsAPI attempts failed - using neutral sentiment.")
    save_file(empty, f"{ticker}_news.csv")
    return empty
