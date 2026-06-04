import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .heston_model import HestonParams, simulate_heston
from .option_pricing import price_european_options
from .config import RISK_FREE_RATE


# Find the most relevant and tradable call options near the current market price
def liquid_strikes(calls: pd.DataFrame, S0: float, max_strikes: int = 15) -> pd.DataFrame:
    # Copy the input DataFrame (to avoid modifying the original)
    df = calls.copy()

    # Compute the mid-price of each option
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # Keep only real tradable options
    df = df[(df["bid"] > 0) & (df["mid"] > 0)]

    # Restrict strikes to a range around the spot price (60% - 140%)
    df = df[(df["strike"] >= S0 * 0.6) & (df["strike"] <= S0 * 1.4)]

    # Compute moneyness (smaller = closer to at-the-money)
    df["moneyness"] = (df["strike"] - S0).abs() / S0

    # Sort by moneyness and reset index for a clean DataFrame
    return df.sort_values("moneyness").head(max_strikes).reset_index(drop=True)


# Calibrate Heston model parameters to market call prices.
def calibrate_heston(calls: pd.DataFrame, S0: float, T: float, theta: float, v0: float, r: float = RISK_FREE_RATE, n_paths: int = 4000, seed: int = 0) -> HestonParams:
    liquid = liquid_strikes(calls, S0)

    # If no usable strikes found - return default parameters
    if liquid.empty:
        print("  [calib] No liquid strikes found — using prior parameters.")
        return HestonParams(theta=theta, v0=v0, r=r)

    # Extract strikes and market mid prices
    strikes = liquid["strike"].values
    market_mids = liquid["mid"].values
    print(f"  [calib] Fitting to {len(strikes)} liquid strikes ...")

    # Objective function — optimises over (kappa, f, rho, v0_cal)
    # sigma_v is derived from kappa and f to enforce the Feller condition
    def objective(x: np.ndarray) -> float:
        kappa, f, rho, v0_cal = x

        # Feller-safe transformation: sigma_v = f * sqrt(2 * kappa * theta)
        # Since f in (0, 1): sigma_v^2 = f^2 * 2*kappa*theta < 2*kappa*theta  (always)
        sigma_v = f * np.sqrt(2.0 * kappa * theta)

        hp = HestonParams(kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, v0=v0_cal, r=r)
        try:
            # Simulate Heston model paths for the underlying asset
            S_paths, _ = simulate_heston(
                S0, T, hp,
                n_paths=n_paths,
                n_steps=max(int(T * 252), 20),
                seed=seed,
            )
            S_T = S_paths[:, -1]

            # Compute model call prices for each strike using Monte Carlo
            model_prices = np.array([
                price_european_options(S_T, K, r, T)["call"]
                for K in strikes
            ])

            # MSE vs market mid prices — no penalty term needed
            return float(np.mean((model_prices - market_mids) ** 2))
        except Exception:
            return 1e10

    # Bounds for (kappa, f, rho, v0)
    v0_bounds = (max(v0 * 0.2, 0.001), min(v0 * 3.0, 0.5))
    bounds = [
        (0.5, 20.0),        # kappa
        (0.01, 0.99),       # f  (Feller fraction)
        (-0.99, 0.0),       # rho
        v0_bounds,          # v0
    ]

    # Run global optimisation (differential evolution) to minimise objective
    result = differential_evolution(
        objective, bounds,
        maxiter=50, popsize=8, tol=1e-4,
        seed=seed, workers=1, disp=False,
    )

    # Unpack and recover sigma_v from the Feller transformation
    kappa_opt, f_opt, rho_opt, v0_opt = result.x
    sigma_v_opt = f_opt * np.sqrt(2.0 * kappa_opt * theta)

    # Feller margin
    feller_margin = 2.0 * kappa_opt * theta - sigma_v_opt ** 2

    print(f"  [calib] kappa={kappa_opt:.3f}  sigma_v={sigma_v_opt:.3f}  "
          f"rho={rho_opt:.3f}  v0={v0_opt:.5f}  MSE={result.fun:.5f}  "
          f"Feller margin={feller_margin:.5f}")

    return HestonParams(
        kappa=kappa_opt, theta=theta, sigma_v=sigma_v_opt,
        rho=rho_opt, v0=v0_opt, r=r,
    )