# Shared utility functions
import datetime
import numpy as np
import pandas as pd

# T = calendar_days_to_expiry / 252 (trading-day convention)
# Floored at 1/252 (one trading day)
def compute_time_to_expiry(expiry_date: datetime.date) -> float:
    days = (expiry_date - datetime.date.today()).days
    return max(days/252, 1/252)

# Annualised realised variance from the last window log-returns
# Floored at 0.001 (approx. 3.2% vol) to avoid degenerate calibration
def estimate_realised_variance(price_series: pd.Series, window: int = 21) -> float:
    log_return = np.log(price_series / price_series.shift(1)).dropna()
    rv = log_return.iloc[-window:].var() * 252
    return max(float(rv), 0.001)

# RiskMetrics Exponentially Weighted Moving Average (EWMA) variance
# sigma^2_t = lam*sigma^2_(t-1) + (1-lam)*r^2_(t-1)
# Lambda = 0.94 -> approx. 17-day effective lookback
def compute_ewma_variance(price_series: pd.Series, lam: float = 0.94) -> float:
    log_return = np.log(price_series / price_series.shift(1)).dropna().values
    var_ewma = float(log_return[0] ** 2)
    for r in log_return[1:]:
        var_ewma = lam * var_ewma + (1 - lam) * r ** 2
    return max(var_ewma * 252, 0.001)