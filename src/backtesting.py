
"""
backtesting.py
Final version – includes 1D sequential optimisation, compatible with main.py and accepts extra kwargs (e.g., top_tickers).
"""

import numpy as np
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm
from backtesting_env import RuleTradingEnv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path("logs").mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Single-ticker optimisation worker
# -----------------------------------------------------------------------------
def optimize_single_ticker_worker(params):
    (
        ticker, train_start, train_end, current_target_perc, current_class_horizon,
        feature_set, model_buy, model_sell, scaler,
        capital_per_stock, current_buy_proba, current_sell_proba,
        force_percentage_optimization, force_thresholds_optimization
    ) = params

    def evaluate_thresholds(p_buy, p_sell, p_target, c_horizon):
        try:
            env = RuleTradingEnv(
                ticker=ticker,
                model_buy=model_buy,
                model_sell=model_sell,
                scaler=scaler,
                capital_per_stock=capital_per_stock,
                train_start=train_start,
                train_end=train_end,
                feature_set=feature_set
            )

            if hasattr(env, "set_thresholds"):
                env.set_thresholds(
                    p_buy=p_buy,
                    p_sell=p_sell,
                    p_target=p_target,
                    c_horizon=c_horizon
                )
            else:
                env.p_buy = p_buy
                env.p_sell = p_sell
                env.p_target = p_target
                env.c_horizon = c_horizon

            result = env.run()
            return result.get("revenue", 0.0)
        except Exception as e:
            logger.error(f"[{ticker}] Evaluation failed for ({p_buy}, {p_sell}, {p_target}, {c_horizon}): {e}")
            return -np.inf

    print(f"\n🔍 [{ticker}] Starting sequential 1D threshold optimisation...")

    best_p_buy = current_buy_proba
    best_p_sell = current_sell_proba
    best_p_target = current_target_perc
    best_c_horizon = current_class_horizon

    # 1️⃣ Optimise p_buy
    best_metric = -np.inf
    for pb in np.arange(0.0, 0.55, 0.05):
        metric = evaluate_thresholds(pb, best_p_sell, best_p_target, best_c_horizon)
        if metric > best_metric:
            best_metric, best_p_buy = metric, pb
    print(f"[{ticker}] ✅ Best p_buy={best_p_buy:.3f}")

    # 2️⃣ Optimise p_sell
    best_metric = -np.inf
    for ps in np.arange(0.0, 0.55, 0.05):
        metric = evaluate_thresholds(best_p_buy, ps, best_p_target, best_c_horizon)
        if metric > best_metric:
            best_metric, best_p_sell = metric, ps
    print(f"[{ticker}] ✅ Best p_sell={best_p_sell:.3f}")

    # 3️⃣ Optimise p_target
    best_metric = -np.inf
    for pt in np.arange(0.002, 0.020, 0.002):
        metric = evaluate_thresholds(best_p_buy, best_p_sell, pt, best_c_horizon)
        if metric > best_metric:
            best_metric, best_p_target = metric, pt
    print(f"[{ticker}] ✅ Best p_target={best_p_target:.4f}")

    # 4️⃣ Optimise c_horizon
    best_metric = -np.inf
    for ch in [1, 3, 5, 10]:
        metric = evaluate_thresholds(best_p_buy, best_p_sell, best_p_target, ch)
        if metric > best_metric:
            best_metric, best_c_horizon = metric, ch
    print(f"[{ticker}] ✅ Best c_horizon={best_c_horizon}")

    return {
        "ticker": ticker,
        "min_proba_buy": best_p_buy,
        "min_proba_sell": best_p_sell,
        "target_percentage": best_p_target,
        "class_horizon": best_c_horizon,
        "optimization_status": "1D-OPTIMIZED"
    }


# -----------------------------------------------------------------------------
# Portfolio-level parallel optimisation
# -----------------------------------------------------------------------------
def optimize_thresholds_for_portfolio_parallel(optimization_params, num_processes=None, **kwargs):
    num_processes = num_processes or max(1, cpu_count() - 1)
    print(f"\n⚙️ Launching parallel 1D optimisation for {len(optimization_params)} tickers "
          f"on {num_processes} CPU cores...")

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(optimize_single_ticker_worker, optimization_params),
                total=len(optimization_params),
                desc="Optimizing Thresholds (1D)"
            )
        )

    optimized_params_per_ticker = {}
    for res in results:
        if res and res["ticker"]:
            optimized_params_per_ticker[res["ticker"]] = {
                "min_proba_buy": res["min_proba_buy"],
                "min_proba_sell": res["min_proba_sell"],
                "target_percentage": res["target_percentage"],
                "class_horizon": res["class_horizon"],
                "optimization_status": res["optimization_status"]
            }
            print(f"✅ {res['ticker']} optimised: "
                  f"Buy={res['min_proba_buy']:.2f}, "
                  f"Sell={res['min_proba_sell']:.2f}, "
                  f"Target={res['target_percentage']:.3%}, "
                  f"CH={res['class_horizon']}")

    return optimized_params_per_ticker


if __name__ == "__main__":
    print("Run from main.py with FORCE_THRESHOLDS_OPTIMIZATION=True to execute optimisation.")
