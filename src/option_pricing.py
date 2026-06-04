# Price European calls and puts from a vector of simulated terminal prices
import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE

# Risk-Neutral Pricing using Monte Carlo.
# C = e^(-rT) * E[max(S_T - K, 0)]
# P = e^(-rT) * E[max(K - S_T, 0)]
def price_european_options(S_T: np.ndarray, 
                           K: float, 
                           r: float = RISK_FREE_RATE, 
                           T: float = 0.25) -> dict[str, float]:
    # Account for the present value of money
    discount = np.exp(-r*T)
    call_pf = np.maximum(S_T - K, 0.0)
    put_pf = np.maximum(K - S_T, 0.0)
    n = len(S_T)
    return {
        "call":     discount * call_pf.mean(),
        "put":      discount * put_pf.mean(),
        "call_std": discount * call_pf.std() / np.sqrt(n),
        "put_std":  discount * put_pf.std()  / np.sqrt(n),
    }

# Price calls and puts for every strike in strikes
def price_option_chain(S_T: np.ndarray,
                       strikes: np.ndarray,
                       r: float = RISK_FREE_RATE,
                       T: float = 0.25) -> pd.DataFrame:
    rows = []
    for K in strikes:
        p = price_european_options(S_T, float(K), r, T)
        rows.append({"strike": K, **p})
    return pd.DataFrame(rows)

# Returns residual
def put_call_parity_residual(S0: float, K: float, r: float, T: float, call: float, put: float) -> float:
    return (call - put) - (S0 - K * np.exp(-r * T))

# Black-Scholes call price (Backup when chain is illiquid)
def black_scholes_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    import math
    from scipy.stats import norm
    if T <= 0 or sigma <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))