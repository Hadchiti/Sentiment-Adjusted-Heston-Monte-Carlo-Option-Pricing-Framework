# Sentiment-Adjusted Heston Monte Carlo Option Pricing Framework
# Full End-to-End Pipeline

import os
import sys
import traceback
import numpy as np
import pandas as pd

from src.config import (
    TICKER, NEWS_API_KEY, HESTON_PARAMS, SENTIMENT_ALPHA,
    RISK_FREE_RATE, N_PATHS, N_STEPS, DATA_DIR,
)

from src.data_fetch import get_stock_prices, get_option_chain, get_news
from src.sentiment_analysis import calculate_daily_sentiment
from src.heston_model import HestonParams, simulate_heston, apply_sentiment_to_theta
from src.calibration import calibrate_heston
from src.option_pricing import price_option_chain, put_call_parity_residual, black_scholes_call
from src.visualise_results import (
    plot_stock_paths,
    plot_variance_paths,
    plot_option_comparison,
    plot_sentiment,
)
from src.utils import compute_time_to_expiry, estimate_realised_variance


# Divide Each Section in the Console
def section(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:

    os.makedirs(DATA_DIR, exist_ok=True)

    print("\nSentiment-Adjusted Heston Monte Carlo Framework")
    print(f"Ticker: {TICKER}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE:.4f}")
    print()

    # STEP 1 — Historical Stock Prices
    section("STEP 1 — Fetch Historical Prices")
    try:
        price_df = get_stock_prices(TICKER)
        S0 = float(price_df["Close"].iloc[-1])
        print(f"Initial price S0 = {S0:.4f}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # STEP 2 — Option Chain
    section("STEP 2 — Fetch Option Chain")
    try:
        calls, puts, expiry_date = get_option_chain(TICKER)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # STEP 3 — Time to Expiry
    section("STEP 3 — Compute Time to Expiry")
    T = compute_time_to_expiry(expiry_date)
    T_calib = max(T, 5 / 252)
    print(f"Expiry: {expiry_date}")
    print(f"T = {T:.6f} years")

    # STEP 4 — News Sentiment
    section("STEP 4 — News Sentiment")
    try:
        news_df = get_news(TICKER, NEWS_API_KEY)
        sentiment = calculate_daily_sentiment(news_df, TICKER)
        current_sentiment = float(sentiment.mean()) if not sentiment.empty else 0.0
    except Exception:
        traceback.print_exc()
        sentiment = pd.Series(dtype=float)
        current_sentiment = 0.0

    # STEP 5 — Initial Variance v0
    section("STEP 5 — Estimate Initial Variance")
    try:
        v0 = estimate_realised_variance(price_df["Close"], window=21)
        print(f"v0 = {v0:.6f}")
    except Exception:
        v0 = HESTON_PARAMS["v0"]
        print(f"Fallback v0 = {v0:.6f}")

    # STEP 6 — Sentiment-Adjusted Long-Run Variance
    section("STEP 6 — Adjust Long-Run Variance")
    theta_base = HESTON_PARAMS["theta"]
    theta_adj = apply_sentiment_to_theta(theta_base, current_sentiment, SENTIMENT_ALPHA)
    print(f"theta_base = {theta_base:.6f}")
    print(f"theta_adj  = {theta_adj:.6f}")

    # STEP 7 — Calibration
    section("STEP 7 — Calibrate Heston Parameters")
    try:
        calibrated_params = calibrate_heston(
            calls=calls,
            S0=S0,
            T=T_calib,
            theta=theta_adj,
            v0=v0,
            r=RISK_FREE_RATE,
            n_paths=10000,
        )
    except Exception:
        traceback.print_exc()
        calibrated_params = HestonParams(
            kappa=HESTON_PARAMS["kappa"],
            theta=theta_adj,
            sigma_v=HESTON_PARAMS["sigma_v"],
            rho=HESTON_PARAMS["rho"],
            v0=v0,
            r=RISK_FREE_RATE,
        )

    print(calibrated_params)

    # STEP 8 — Monte Carlo Simulation
    section("STEP 8 — Simulate Heston Paths")
    try:
        S_paths, v_paths = simulate_heston(
            S0,
            T_calib,
            calibrated_params,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            seed=42,
        )
        S_T = S_paths[:, -1]
        print(f"Mean terminal price = {S_T.mean():.4f}")
        print(f"Std terminal price  = {S_T.std():.4f}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # STEP 9 — Price Options
    section("STEP 9 — Price Options")

    try:
        market_strikes = calls["strike"].values
        model_df = price_option_chain(S_T, market_strikes, RISK_FREE_RATE, T_calib)

        # Rename columns consistently
        if "call" in model_df.columns:
            model_df = model_df.rename(columns={"call": "model_call"})
        if "put" in model_df.columns:
            model_df = model_df.rename(columns={"put": "model_put"})

        # Save full model prices
        model_df.to_csv(os.path.join(DATA_DIR, f"{TICKER}_model_prices.csv"), index=False)

        # ATM index
        atm_idx = int((model_df["strike"] - S0).abs().idxmin())
        atm = model_df.iloc[atm_idx]

        if "model_call" in atm and "model_put" in atm:
            residual = put_call_parity_residual(
                S0,
                atm["strike"],
                RISK_FREE_RATE,
                T_calib,
                atm["model_call"],
                atm["model_put"],
            )
            print(f"Put-call parity residual = {residual:.6f}\n")

        # Display a small table around ATM (+-2 strikes if possible)
        start_idx = max(atm_idx - 2, 0)
        end_idx = min(atm_idx + 3, len(model_df))
        atm_slice = model_df.iloc[start_idx:end_idx]

        print("Sample of Heston Model Option Prices (Closest to ATM):")
        print(atm_slice[["strike", "model_call", "model_put"]].to_string(index=False))

        # Ensure market calls have usable bids for comparison plots
        calls_liquid = calls[calls["bid"] > 0].copy()
        if calls_liquid.empty:
            sigma_proxy = np.sqrt(v0)
            calls_liquid = calls.copy()
            calls_liquid["bid"] = calls_liquid["strike"].apply(
                lambda K: black_scholes_call(S0, K, RISK_FREE_RATE, T_calib, sigma_proxy) * 0.98
            )
            calls_liquid["ask"] = calls_liquid["strike"].apply(
                lambda K: black_scholes_call(S0, K, RISK_FREE_RATE, T_calib, sigma_proxy) * 1.02
            )

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # STEP 10 — Visualisation
    section("STEP 10 — Generate Visualisations")
    try:
        plot_stock_paths(S_paths, T_calib, S0, TICKER)
        plot_variance_paths(v_paths, T_calib, calibrated_params.theta, TICKER)
        plot_option_comparison(model_df, calls_liquid, TICKER)

        if not sentiment.empty:
            plot_sentiment(sentiment, TICKER)

    except Exception as exc:
        print(f"Visualisation error: {exc}")

    print("\nPipeline complete.")
    print(f"Outputs saved to: {DATA_DIR}\n")


if __name__ == "__main__":
    main()