# Heston Stochastic Volatility Model (1993)
# Vectorised Euler-Maruyama Simulation (To solve the SDE)
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import (RISK_FREE_RATE, N_PATHS, N_STEPS, SENTIMENT_ALPHA)

# All parameters needed to define and simmulate a Heston model
@dataclass 
class HestonParams:
    kappa:   float = 2.0                # mean-reversion speed
    theta:   float = 0.04               # long-run variance
    sigma_v: float = 0.3                # vol-of-vol
    rho:     float = -0.7               # price–variance correlation
    v0:      float = 0.04               # initial variance
    r:       float = RISK_FREE_RATE     # risk-free rate


# This function simulates the Heston stochastic volatility model.
# It uses a vectorised Monte-Carlo approach.
# The Euler–Maruyama method is used to step forward in small time increments.

def simulate_heston(S0: float, 
                    T: float, 
                    params: HestonParams, 
                    n_paths: int = N_PATHS, 
                    n_steps: int = N_STEPS,
                    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    
    # Random number generator
    rng = np.random.default_rng(seed)

    # Unpack the Heston parameters
    kappa, theta, sigma_v, rho, v0, r = (
        params.kappa, params.theta, params.sigma_v,
        params.rho, params.v0, params.r,
    )

    # Time step size
    dt = T / n_steps

    # Constants
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(max(1.0-rho**2, 0.0))

    # Array for Monte Carlo Paths
    S = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    # Initial values at t = 0
    S[:, 0] = S0
    v[:, 0] = v0

    # Time-stepping loop for Euler-Maruyama discretisation
    for t in range(n_steps):

        # Generate independent standard normal shocks
        W1 = rng.standard_normal(n_paths)
        W2 = rng.standard_normal(n_paths)

        # Create correlated Brownian increments using Cholesky transformation
        Z1 = W1
        Z2 = rho * W1 + sqrt_1_rho2 * W2

        # Ensure non-negative variance - then square root
        v_cur = np.maximum(v[:, t], 0.0)
        sv_cur = np.sqrt(v_cur)

        # Variance update
        # dv_t = kappa*(theta - v_t)*dt + sigma_v*sqrt(v_t)*dW2
        v[:, t+1] = (v_cur + kappa * (theta - v_cur) * dt + sigma_v * sv_cur * sqrt_dt * Z2)

        # Stock Price Update - log form for stability (prevent negative prices)
        # dS_t = r S_t dt + sqrt(v_t) S_t dW1
        log_S = np.log(S[:, t]) 
        S[:, t+1] = np.exp(log_S + (r - 0.5 * v_cur) * dt + sv_cur * sqrt_dt * Z1)

    # Return full simulated paths
    return S, v

# Return sentiment adjusted long-run variance:
# theta_adj = theta_base + alpha * sentiment
def apply_sentiment_to_theta(theta_base: float, sentiment: float, alpha: float = SENTIMENT_ALPHA) -> float:
    # Ensure variance strictly positive
    return max(theta_base + alpha * sentiment, 1e-6)