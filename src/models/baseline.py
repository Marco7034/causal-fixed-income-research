"""
Out-of-Sample Baseline vs. Causal-Gated Forecasting Benchmarks.

Evaluates whether pruning the input feature space to verified PCMCI+ causal parents
improves out-of-sample credit spread forecasting across major
macroeconomic regime shifts.

Benchmarks evaluated on both:
  1. Macro Aggregate Credit Spread (`d_CS_BBB`: 2010-2019 ZIRP -> 2022-2026 Tightening)
  2. Micro Bond Panel (`d_CS_i`: 2019-2021 Low-Rate -> 2022-2023 Tightening across 160 bonds)

Models compared:
  - Model 1: Random Walk (Zero Spread Change Benchmark)
  - Model 2: Autoregressive Baseline AR(L)
  - Model 3: Unconstrained Kitchen-Sink (All Lagged Shocks + Levels)
  - Model 4: Unconstrained Lagged Shocks (All Candidate Variables x All Lags)
  - Model 5: Causal-Gated Model (Strictly Pruned to PCMCI+ Verified Lagged Parents)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "causal_artifacts"
FIGURES_DIR = PROJECT_ROOT / "figures"


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_candidate: np.ndarray,
    y_pred_benchmark: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes the Diebold-Mariano (1995) test statistic for equal squared forecast error.
    A positive DM statistic indicates `y_pred_candidate` has strictly lower MSE than
    `y_pred_benchmark`.
    """
    e_cand = (y_true - y_pred_candidate) ** 2
    e_bench = (y_true - y_pred_benchmark) ** 2
    d = e_bench - e_cand  # Positive when candidate beats benchmark
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    n = len(d)
    if std_d < 1e-15 or n < 2:
        return 0.0, 1.0
    dm_stat = mean_d / (std_d / np.sqrt(n))
    p_val = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_val


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bench_shocks: np.ndarray,
    model_name: str,
    n_features: int,
) -> Dict:
    """Computes out-of-sample RMSE, MAE, OOS R^2, IC, Directional Hit Rate, and DM stat."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2_oos = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2))

    if np.std(y_pred) > 1e-12:
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        ic = 0.0

    nz = np.abs(y_true) > 1e-6
    if np.std(y_pred) > 1e-12 and np.any(nz):
        dir_hit = float(np.mean(np.sign(y_true[nz]) == np.sign(y_pred[nz])) * 100.0)
    else:
        dir_hit = 50.0

    dm_stat, dm_pval = diebold_mariano_test(y_true, y_pred, y_bench_shocks)

    return {
        "Model": model_name,
        "Features": n_features,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "OOS_R2_pct": round(r2_oos * 100.0, 2),
        "IC": round(ic, 4),
        "Dir_Hit_pct": round(dir_hit, 1),
        "DM_vs_Unconstrained": round(dm_stat, 2),
        "DM_pval": round(dm_pval, 5),
    }


def run_macro_forecast_benchmark(
    stat_path: Path = PROJECT_ROOT / "data" / "processed" / "data_stationary_2000_2026.csv",
    lvl_path: Path = PROJECT_ROOT / "data" / "raw" / "data_levels_2000_2026.csv",
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Macro Out-of-Sample Forecast Evaluation.

    Trains strictly on the ZIRP regime (2010-2019) and evaluates out-of-sample on
    the Post-COVID Inflation & Tightening regime (2022-2026) using only lagged features.
    """
    df_stat = pd.read_csv(stat_path, index_col=0, parse_dates=True).sort_index()
    df_lvl = pd.read_csv(lvl_path, index_col=0, parse_dates=True).sort_index()

    all_shock_vars = ["d_CS_BBB", "d_DGS10", "d_T10Y2Y", "ret_SPY", "d_VIX", "d_DGS10_sq"]
    feat_dict = {}
    for v in all_shock_vars:
        for lag in range(1, max_lag + 1):
            feat_dict[f"{v}_lag{lag}"] = df_stat[v].shift(lag)

    lvl_vars = ["DGS10", "T10Y2Y", "VIX", "DFF"]
    for v in lvl_vars:
        feat_dict[f"{v}_lvl_lag1"] = df_lvl[v].shift(1)

    data = pd.DataFrame(feat_dict)
    data["target"] = df_stat["d_CS_BBB"]
    data = data.dropna()

    train = data.loc["2010-01-01":"2019-12-31"]
    test = data.loc["2022-01-01":"2026-10-01"]

    y_tr = train["target"].values
    y_te = test["target"].values

    ar_cols = [f"d_CS_BBB_lag{l}" for l in range(1, max_lag + 1)]
    shk_cols = [f"{v}_lag{l}" for v in all_shock_vars for l in range(1, max_lag + 1)]
    ks_cols = [c for c in data.columns if c != "target"]
    # Verified STABLE_CORE lagged parents from Phase 2 PCMCI+
    causal_cols = ["ret_SPY_lag1", "ret_SPY_lag2"]

    def fit_predict(cols: List[str]) -> np.ndarray:
        w = np.linalg.lstsq(train[cols].values, y_tr, rcond=None)[0]
        return test[cols].values @ w

    pred_rw = np.zeros_like(y_te)
    pred_ar = fit_predict(ar_cols)
    pred_ks = fit_predict(ks_cols)
    pred_shk = fit_predict(shk_cols)
    pred_cg = fit_predict(causal_cols)

    rows = [
        evaluate_predictions(y_te, pred_rw, pred_shk, "1. Random Walk (Zero Change)", 0),
        evaluate_predictions(y_te, pred_ar, pred_shk, "2. Autoregressive AR(5)", len(ar_cols)),
        evaluate_predictions(y_te, pred_ks, pred_shk, "3. Unconstrained Kitchen-Sink (Lags + Levels)", len(ks_cols)),
        evaluate_predictions(y_te, pred_shk, pred_shk, "4. Unconstrained All Lagged Shocks", len(shk_cols)),
        evaluate_predictions(y_te, pred_cg, pred_shk, "5. Causal-Gated Model (PCMCI+ Core Lags)", len(causal_cols)),
    ]
    return pd.DataFrame(rows)


def run_panel_forecast_benchmark(
    panel_path: Path = PROJECT_ROOT / "data" / "processed" / "MASTER_MODEL_PANEL.csv",
    max_lag: int = 3,
) -> pd.DataFrame:
    """
    Micro Bond Panel Out-of-Sample Forecast Evaluation (160 Corporate Bonds).

    Trains bond-specific WLS models on the 2019-2021
    Low-Rate era, and tests strictly out-of-sample on the 2022-2023 Tightening era.
    """
    df = pd.read_csv(panel_path, parse_dates=["date"])
    df = df.sort_values(["cusip", "date"]).reset_index(drop=True)

    df["log_vol"] = np.log1p(df["daily_volume"].clip(lower=0.0))
    df["d_log_vol"] = df.groupby("cusip")["log_vol"].diff()

    # Strictly causal trailing 20-day volatility weight (uses only <= t-1)
    df["roll_std"] = (
        df.groupby("cusip")["d_CS_i"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
        .clip(lower=0.01)
    )

    lag_vars = ["d_CS_i", "d_DGS10", "d_T10Y2Y", "d_VIX", "daily_return", "d_log_vol"]
    for v in lag_vars:
        for l in range(1, max_lag + 1):
            df[f"{v}_lag{l}"] = df.groupby("cusip")[v].shift(l)

    lvl_vars = ["leverage", "ttm_years", "CS_i_raw10y"]
    for v in lvl_vars:
        df[f"{v}_lag1"] = df.groupby("cusip")[v].shift(1)

    ar_cols = [f"d_CS_i_lag{l}" for l in range(1, max_lag + 1)]
    shk_cols = [f"{v}_lag{l}" for v in lag_vars for l in range(1, max_lag + 1)]
    ks_cols = shk_cols + [f"{v}_lag1" for v in lvl_vars]
    # Verified lagged causal parents from Phase 3 Multi-Entity PCMCI+
    causal_cols = [
        "d_CS_i_lag1",
        "d_CS_i_lag2",
        "d_DGS10_lag1",
        "d_DGS10_lag2",
        "d_VIX_lag2",
        "d_VIX_lag3",
        "daily_return_lag1",
    ]

    df = df.dropna(subset=ks_cols + ["d_CS_i", "roll_std"]).reset_index(drop=True)

    preds = {k: [] for k in ["y_true", "rw", "ar", "ks", "shk", "cg"]}
    n_bonds_evaluated = 0

    for _, grp in df.groupby("cusip"):
        tr = grp[grp["date"] < "2022-01-01"]
        te = grp[grp["date"] >= "2022-01-01"]
        if len(tr) < 100 or len(te) < 50:
            continue
        n_bonds_evaluated += 1
        sw = 1.0 / tr["roll_std"].values
        y_tr = tr["d_CS_i"].values
        y_te = te["d_CS_i"].values

        preds["y_true"].append(y_te)
        preds["rw"].append(np.zeros_like(y_te))

        for key, cols in [("ar", ar_cols), ("ks", ks_cols), ("shk", shk_cols), ("cg", causal_cols)]:
            X_tr = np.column_stack([np.ones(len(tr)), tr[cols].values])
            X_te = np.column_stack([np.ones(len(te)), te[cols].values])
            w = np.linalg.lstsq(X_tr * sw[:, None], y_tr * sw, rcond=None)[0]
            preds[key].append(X_te @ w)

    y_true = np.concatenate(preds["y_true"])
    p_rw = np.concatenate(preds["rw"])
    p_ar = np.concatenate(preds["ar"])
    p_ks = np.concatenate(preds["ks"])
    p_shk = np.concatenate(preds["shk"])
    p_cg = np.concatenate(preds["cg"])

    logger.info(f"Evaluated bond-specific out-of-sample models across {n_bonds_evaluated} bonds ({len(y_true):,} test obs).")

    rows = [
        evaluate_predictions(y_true, p_rw, p_shk, "1. Random Walk (Zero Change)", 0),
        evaluate_predictions(y_true, p_ar, p_shk, "2. Autoregressive AR(3)", len(ar_cols)),
        evaluate_predictions(y_true, p_ks, p_shk, "3. Unconstrained Kitchen-Sink (Lags + Levels)", len(ks_cols)),
        evaluate_predictions(y_true, p_shk, p_shk, "4. Unconstrained All Lagged Shocks", len(shk_cols)),
        evaluate_predictions(y_true, p_cg, p_shk, "5. Causal-Gated Model (7 PCMCI+ Parents)", len(causal_cols)),
    ]
    return pd.DataFrame(rows)


def plot_benchmark_results(
    df_macro: pd.DataFrame,
    df_panel: pd.DataFrame,
    fig_dir: Path = FIGURES_DIR,
):
    """Generates a visual comparison of Out-of-Sample R^2 across baseline models."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    short_labels = [
        "Random Walk",
        "AR Baseline",
        "Kitchen-Sink\n(Lags+Levels)",
        "Unconstrained\nLagged Shocks",
        "Causal-Gated\n(PCMCI+ Pruned)",
    ]
    colors = ["#7f7f7f", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    # Panel A: Macro OOS R^2 (%)
    ax1 = axes[0]
    vals1 = df_macro["OOS_R2_pct"].values
    bars1 = ax1.bar(short_labels, vals1, color=colors, alpha=0.88, width=0.55)
    ax1.axhline(0, color="black", lw=0.9, linestyle="--")
    ax1.set_ylabel("Out-of-Sample R² (%)")
    ax1.set_title("A. Macro Credit Spread Forecast (Train: ZIRP -> Test: Tightening)")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars1, vals1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.08 if v >= 0 else -0.22),
            f"{v:+.2f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    # Panel B: Micro Panel OOS R^2 (%)
    ax2 = axes[1]
    vals2 = df_panel["OOS_R2_pct"].values
    bars2 = ax2.bar(short_labels, vals2, color=colors, alpha=0.88, width=0.55)
    ax2.axhline(0, color="black", lw=0.9, linestyle="--")
    ax2.set_ylabel("Out-of-Sample R² (%)")
    ax2.set_title("B. Micro Bond Panel Forecast (160 Bonds, N_test=77,698)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, vals2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.25 if v >= 0 else -0.65),
            f"{v:+.2f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = fig_dir / "baseline_forecast_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved baseline forecast comparison figure to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df_macro_res = run_macro_forecast_benchmark()
    df_panel_res = run_panel_forecast_benchmark()

    df_macro_res.to_csv(ARTIFACTS_DIR / "macro_forecast_benchmark.csv", index=False)
    df_panel_res.to_csv(ARTIFACTS_DIR / "panel_forecast_benchmark.csv", index=False)
    plot_benchmark_results(df_macro_res, df_panel_res)

    print("\n" + "=" * 105)
    print(" TRACK 1: MACRO OUT-OF-SAMPLE FORECAST BENCHMARK (Train: ZIRP 2010-2019 -> Test: Tightening 2022-2026)")
    print("=" * 105)
    print(df_macro_res.to_string(index=False))

    print("\n" + "=" * 105)
    print(" TRACK 2: MICRO PANEL OUT-OF-SAMPLE FORECAST BENCHMARK (Train: 2019-2021 -> Test: 2022-2023, 160 Bonds)")
    print("=" * 105)
    print(df_panel_res.to_string(index=False))
    print("=" * 105 + "\n")
