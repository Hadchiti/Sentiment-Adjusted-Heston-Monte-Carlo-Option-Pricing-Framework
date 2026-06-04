# Heston-Project
This repository contains a Sentiment-Adjusted Heston Monte Carlo Option Pricing Framework.

# Project Structure
├── main.py                  # End-to-end pipeline entry point
├── src/
│   ├── config.py            # All user settings (ticker, parameters, API key)
│   ├── data_fetch.py        # Yahoo Finance prices + option chain; NewsAPI headlines
│   ├── sentiment_analysis.py# TextBlob polarity scoring; daily aggregation
│   ├── heston_model.py      # HestonParams dataclass; Euler-Maruyama simulator
│   ├── calibration.py       # Differential Evolution calibration; Feller reparameterisation
│   ├── option_pricing.py    # MC option pricing; put-call parity; Black-Scholes benchmark
│   ├── utils.py             # Time-to-expiry; realised variance; EWMA variance
│   └── visualise_results.py # Four diagnostic matplotlib plots
└── data/                    # Auto-created; all CSV outputs and PNG plots saved here

# Installation
pip install -r requirements.txt
python -m textblob.download_corpora

# Configuration
Set your NewsAPI key in a .env file in the project root:
NEWS_API_KEY=your_key_here

All other settings (ticker, risk-free rate, Monte Carlo parameters) are in src/config.py.

# Usage
python main.py
