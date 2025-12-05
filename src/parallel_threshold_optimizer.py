
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import json
from backtesting_env import RuleTradingEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path("logs").mkdir(exist_ok=True)

# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate_thresholds(ticker, p_buy, p_sell, p_target, c_horizon):
    """Run a single backtest with given parameters and return performance metric."""
    try:
        env = RuleTradingEnv(
            ticker=ticker,
            p_buy=p_buy,
            p_sell=p_sell,
            p_target=p_target,
            c_horizon=c_horizon
        )
        result = env.run()
        metric = result.get("revenue", 0.0)
        logger.debug(
            f"[EVAL] {ticker}: Buy={p_buy:.3f}, Sell={p_sell:.3f}, Target={p_target:.4f}, CH={c_horizon} -> {metric:.2f}"
        )
        return metric
    except Exception as e:
        logger.error(f"[EVAL-ERR] {ticker}: Failed evaluation ({e})")
        return -np.inf


# ------------------------------
# 1D Optimisation Helper
# ------------------------------
def optimise_threshold_1d(ticker, grid, param_name, defaults):
    """Perform 1D sweep for a single parameter."""
    best_val, best_metric = None, -np.inf

    for val in grid:
        params = defaults.copy()
        params[param_name] = val
        metric = evaluate_thresholds(ticker, **params)

        if metric > best_metric:
            best_metric = metric
            best_val = val

        logger.info(f"[1D-OPT] {ticker}: {param_name}={val:.4f} -> Metric={metric:.2f}")

    logger.info(f"[1D-OPT] ✅ Best {param_name}={best_val:.4f} (Metric={best_metric:.2f})")
    return best_val


# ------------------------------
# Sequential Multi-Stage 1D Optimisation
# ------------------------------
def optimise_all_thresholds(ticker):
    """Sequentially optimise each threshold parameter independently in 1D sweeps."""
    defaults = {
        "p_buy": 0.05,
        "p_sell": 0.05,
        "p_target": 0.005,
        "c_horizon": 1
    }

    logger.info(f"[1D-OPT] Starting sequential threshold optimisation for {ticker}")

    # --- Optimise p_buy ---
    buy_grid = np.arange(0.00, 0.55, 0.05)
    best_buy = optimise_threshold_1d(ticker, buy_grid, "p_buy", defaults)
    defaults["p_buy"] = best_buy

    # --- Optimise p_sell ---
    sell_grid = np.arange(0.00, 0.55, 0.05)
    best_sell = optimise_threshold_1d(ticker, sell_grid, "p_sell", defaults)
    defaults["p_sell"] = best_sell

    # --- Optimise p_target ---
    target_grid = np.arange(0.002, 0.020, 0.002)
    best_target = optimise_threshold_1d(ticker, target_grid, "p_target", defaults)
    defaults["p_target"] = best_target

    # --- Optimise c_horizon ---
    horizon_grid = [1, 3, 5, 10]
    best_ch = optimise_threshold_1d(ticker, horizon_grid, "c_horizon", defaults)
    defaults["c_horizon"] = best_ch

    logger.info(
        f"[FINAL] {ticker}: Optimal thresholds -> "
        f"Buy={best_buy:.3f}, Sell={best_sell:.3f}, Target={best_target:.4f}, CH={best_ch}"
    )

    return defaults


# ------------------------------
# Parallel Runner
# ------------------------------
def optimise_all_tickers_parallel(tickers, n_jobs=None):
    """Run optimisation for multiple tickers in parallel."""
    n_jobs = n_jobs or max(1, cpu_count() - 1)
    logger.info(f"[PARALLEL] Launching optimisation on {n_jobs} workers for {len(tickers)} tickers")

    with Pool(processes=n_jobs) as pool:
        results = pool.map(optimise_all_thresholds, tickers)

    result_dict = {t: r for t, r in zip(tickers, results)}

    out_path = Path("logs/best_thresholds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"[PARALLEL] ✅ All optimisations completed. Results saved to {out_path}")
    return result_dict


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "PLTR", "MU"]
    optimise_all_tickers_parallel(tickers)
