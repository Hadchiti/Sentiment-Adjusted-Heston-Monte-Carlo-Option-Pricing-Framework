import os

# Load the .env file if present
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.isfile(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

######################################
## USER SETTINGS - User can edit these
######################################

# Stock ticker
TICKER: str = "AAPL"

# Get a key at https://newsapi.org/register
# Make sure to set NEWS_API_KEY in your .env file or OS environment
# Fallback Default Key: "YOUR_NEWSAPI_KEY_HERE"
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY_HERE")

# Data Window
HISTORICAL_PERIOD: str = "1y"
NEWS_LOOKBACK_DAYS: int = 20    # max 20 days for free NewsAPI

# Heston Parameters (Before Calibration)
HESTON_PARAMS: dict = {
    "kappa":   2.0,
    "theta":   0.04,
    "sigma_v": 0.3,
    "rho":     -0.7,
    "v0":      0.04,
} 

# Sentiment Adjustment
# theta_adj = theta_base + SENTIMENT_ALPHA * polarity
# Negative alpha = Negative news polarity = Higher long-run variance
SENTIMENT_ALPHA: float = -0.02

# Monte Carlo
N_PATHS: int = 20000
N_STEPS: int = 252

# Risk-Free Rate
RISK_FREE_RATE: float = 0.048       # Current US risk-free rate (approximately)

# Output
DATA_DIR: str = "data"