# All matplotlib visualisations
# Saves PNG files to DATA_DIR
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import DATA_DIR, TICKER

def save_figure(fig: plt.Figure, filename: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# Stock price paths
def plot_stock_paths(S_paths: np.ndarray, T: float, S0: float, ticker: str = TICKER, n_display: int = 200) -> None:
    t_grid = np.linspace(0, T, S_paths.shape[1])
    sample = S_paths[:n_display]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_grid, sample.T, lw=0.4, alpha=0.25, color="steelblue")
    ax.plot(t_grid, np.median(S_paths, axis=0), lw=2, color="navy",    label="Median")
    ax.plot(t_grid, np.percentile(S_paths,  5, axis=0), lw=1.5, ls="--",
            color="crimson", label="5th pct")
    ax.plot(t_grid, np.percentile(S_paths, 95, axis=0), lw=1.5, ls="--",
            color="green",   label="95th pct")
    ax.axhline(S0, color="black", lw=1, ls=":", label=f"S₀={S0:.2f}")
    ax.set_title(f"Heston MC — {ticker} Simulated Stock Paths "
                 f"({n_display}/{S_paths.shape[0]} shown)", fontsize=13)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{ticker}_stock_paths.png")

# Variance paths
def plot_variance_paths(v_paths: np.ndarray, T: float, theta: float, ticker: str = TICKER, n_display: int = 200) -> None:
    t_grid = np.linspace(0, T, v_paths.shape[1])
    sample = v_paths[:n_display]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_grid, sample.T, lw=0.4, alpha=0.25, color="darkorange")
    ax.plot(t_grid, np.median(v_paths, axis=0), lw=2, color="saddlebrown",
            label="Median variance")
    ax.axhline(theta, color="black", lw=1.5, ls="--",
               label=f"θ = {theta:.4f}  (long-run variance)")
    ax.set_title(f"Heston MC — {ticker} Variance Paths "
                 f"({n_display}/{v_paths.shape[0]} shown)", fontsize=13)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Instantaneous Variance vₜ")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{ticker}_variance_paths.png")

# Model Vs Market Prices
def plot_option_comparison(model_df: pd.DataFrame, market_calls: pd.DataFrame, ticker: str = TICKER) -> None:
    mkt = market_calls.copy()
    mkt["mid"] = (mkt["bid"] + mkt["ask"]) / 2.0
    mkt = mkt[(mkt["bid"] > 0) & (mkt["mid"] > 0)]
    merged = pd.merge(model_df, mkt[["strike", "mid"]], on="strike", how="inner")

    if merged.empty:
        print("  [viz] No overlapping strikes — skipping option comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged["strike"], merged["model_call"], "o-", color="steelblue",
            lw=2, label="Heston MC")
    ax.fill_between(merged["strike"],
                    merged["model_call"] - 1.96 * merged["call_std"],
                    merged["model_call"] + 1.96 * merged["call_std"],
                    alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(merged["strike"], merged["mid"], "s--", color="crimson",
            lw=2, label="Market mid")
    ax.set_title(f"{ticker} — Heston MC vs Market Call Prices", fontsize=13)
    ax.set_xlabel("Strike ($)"); ax.set_ylabel("Call Price ($)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{ticker}_option_comparison.png")

# Sentiment
def plot_sentiment(sentiment: pd.Series, ticker: str = TICKER) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["crimson" if v < 0 else "seagreen" for v in sentiment.values]
    ax.bar(sentiment.index, sentiment.values, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=1, ls="--")
    if len(sentiment) >= 7:
        roll = sentiment.rolling(7, min_periods=1).mean()
        ax.plot(roll.index, roll.values, color="navy", lw=2,
                label="7-day rolling mean")
        ax.legend(fontsize=9)
    ax.set_title(f"{ticker} — Daily News Sentiment (TextBlob Polarity)", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Polarity [-1, +1]")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, f"{ticker}_sentiment.png")
    