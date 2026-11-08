"""
Tests the Collin-Dufresne, Goldstein, and Martin (2001) & Merton (1974) structural
hypotheses on the 193k-row maturity-matched bond panel (MASTER_MODEL_PANEL.csv):

  1. Pooled Multi-Entity PCMCI+ (`analysis_mode='multiple'`):
     Treats each bond CUSIP as an independent trajectory so lags never cross bond
     boundaries. Evaluates whether firm-level equity returns (`daily_return`) and
     bond trading liquidity shocks (`d_log_vol`) retain direct causal links to
     individual bond spreads (`d_CS_i`) after conditioning on macro shocks
     (`d_DGS10`, `d_T10Y2Y`, `d_VIX`).

  2. Regime Split on the Micro Panel:
     Compares the Low-Rate / COVID era (2019-2021) vs. the Fed Tightening era (2022-2023).

  3. Cross-Sectional Merton Stratification (High vs. Low Leverage & Short vs. Long TTM):
     Tests Merton's nonlinear comparative statics: firms closer to the default boundary
     (higher book leverage) should exhibit stronger causal sensitivity to equity and
     volatility shocks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from .engine import CausalEngine, CausalResult
from .priors import EXPECTED_SIGNS, build_link_assumptions

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "MASTER_MODEL_PANEL.csv"

PANEL_CAUSAL_VARS = [
    "d_CS_i",        # Target: maturity-matched individual bond credit spread change
    "d_DGS10",       # Macro: 10Y Treasury yield shock
    "d_T10Y2Y",      # Macro: 10Y-2Y yield curve slope shock
    "d_VIX",         # Macro: implied volatility shock
    "daily_return",  # Micro (Merton): issuer daily CRSP equity return
    "d_log_vol",     # Micro (Liquidity): log change in daily bond TRACE volume
]


@dataclass
class PanelDiscoveryResult:
    """Stores pooled panel PCMCI+ results across slices."""
    label: str
    n_bonds: int
    n_total_obs: int
    causal_result: CausalResult


class PanelCausalDiscovery:
    """
    Executes multi-entity PCMCI+ causal discovery on MASTER_MODEL_PANEL.csv.
    """

    def __init__(
        self,
        var_names: Optional[List[str]] = None,
        target_var: str = "d_CS_i",
        tau_max: int = 3,
        pc_alpha: float = 0.01,
        min_bond_obs: int = 150,
    ):
        self.var_names = var_names or list(PANEL_CAUSAL_VARS)
        self.target_var = target_var
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.min_bond_obs = min_bond_obs

    @staticmethod
    def load_and_prepare_panel(path: Path = DEFAULT_PANEL_PATH) -> pd.DataFrame:
        """Loads MASTER_MODEL_PANEL.csv and engineers stationary log-volume shocks."""
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values(["cusip", "date"]).reset_index(drop=True)

        # Stationary bond-level liquidity shock: first difference of log(daily_volume)
        df["log_vol"] = np.log1p(df["daily_volume"].clip(lower=0.0))
        df["d_log_vol"] = df.groupby("cusip")["log_vol"].diff()

        # Update EXPECTED_SIGNS for d_log_vol if not present
        EXPECTED_SIGNS.setdefault("d_log_vol", "-")

        df = df.dropna(subset=PANEL_CAUSAL_VARS).reset_index(drop=True)
        return df

    def _build_multi_tigramite_df(
        self, df_slice: pd.DataFrame
    ) -> Tuple[pp.DataFrame, int, int]:
        """
        Converts a panel DataFrame into Tigramite's `analysis_mode='multiple'` DataFrame
        dictionary `{cusip: np.ndarray(T_i, N)}` so lags never cross bond boundaries.
        """
        data_dict: Dict[str, np.ndarray] = {}
        total_obs = 0

        for cusip, grp in df_slice.groupby("cusip", sort=False):
            grp_clean = grp.sort_values("date")[self.var_names].dropna()
            if len(grp_clean) >= self.min_bond_obs:
                arr = grp_clean.values.astype(np.float64)
                # Guard against zero-variance columns within a slice
                if np.all(np.std(arr, axis=0) > 1e-10):
                    data_dict[str(cusip)] = arr
                    total_obs += len(arr)

        if not data_dict:
            raise ValueError("No bonds met the minimum observation threshold in slice.")

        tig_df = pp.DataFrame(
            data=data_dict,
            var_names=self.var_names,
            analysis_mode="multiple",
        )
        return tig_df, len(data_dict), total_obs

    def run_pooled_slice(
        self, df_slice: pd.DataFrame, label: str
    ) -> PanelDiscoveryResult:
        """Runs pooled multi-dataset PCMCI+ on a panel slice."""
        tig_df, n_bonds, total_obs = self._build_multi_tigramite_df(df_slice)

        assumptions = build_link_assumptions(
            self.var_names, target_var=self.target_var, tau_max=self.tau_max
        )

        pcmci = PCMCI(
            dataframe=tig_df,
            cond_ind_test=ParCorr(significance="analytic"),
            verbosity=0,
        )

        logger.info(
            f"[{label}] Running Multi-Entity PCMCI+ across {n_bonds} bonds "
            f"({total_obs:,} bond-day obs), tau_max={self.tau_max}, alpha={self.pc_alpha}"
        )

        results = pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
            link_assumptions=assumptions,
        )

        parents_df = CausalEngine.extract_target_parents(
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            graph=results["graph"],
            var_names=self.var_names,
            target_var=self.target_var,
            alpha_level=self.pc_alpha,
        )

        c_res = CausalResult(
            var_names=self.var_names,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            graph=results["graph"],
            parents_df=parents_df,
        )

        return PanelDiscoveryResult(
            label=label,
            n_bonds=n_bonds,
            n_total_obs=total_obs,
            causal_result=c_res,
        )

    def run_regime_and_merton_splits(
        self, df: pd.DataFrame
    ) -> Dict[str, PanelDiscoveryResult]:
        """
        Runs pooled panel discovery across:
          1. FULL_PANEL (2019-2023)
          2. LOW_RATE_ERA (2019-2021)
          3. TIGHTENING_ERA (2022-2023)
          4. LOW_LEVERAGE (below-median issuer leverage)
          5. HIGH_LEVERAGE (above-median issuer leverage)
        """
        runs: Dict[str, PanelDiscoveryResult] = {}

        # 1. Full panel
        runs["FULL_PANEL"] = self.run_pooled_slice(df, "FULL_PANEL (2019-2023)")

        # 2. Regime splits
        df_low_rate = df[df["date"] < "2022-01-01"]
        df_tight = df[df["date"] >= "2022-01-01"]
        runs["LOW_RATE_ERA"] = self.run_pooled_slice(df_low_rate, "LOW_RATE_ERA (2019-2021)")
        runs["TIGHTENING_ERA"] = self.run_pooled_slice(df_tight, "TIGHTENING_ERA (2022-2023)")

        # 3. Cross-sectional Merton stratification by bond-level mean leverage
        bond_lev = df.groupby("cusip")["leverage"].mean()
        med_lev = float(bond_lev.median())
        low_lev_cusips = set(bond_lev[bond_lev <= med_lev].index)
        high_lev_cusips = set(bond_lev[bond_lev > med_lev].index)

        runs["LOW_LEVERAGE"] = self.run_pooled_slice(
            df[df["cusip"].isin(low_lev_cusips)],
            f"LOW_LEVERAGE (<= {med_lev:.2f})",
        )
        runs["HIGH_LEVERAGE"] = self.run_pooled_slice(
            df[df["cusip"].isin(high_lev_cusips)],
            f"HIGH_LEVERAGE (> {med_lev:.2f})",
        )

        return runs

    @staticmethod
    def build_summary_table(runs: Dict[str, PanelDiscoveryResult]) -> pd.DataFrame:
        """Builds a unified comparison matrix across all panel slices."""
        first_res = next(iter(runs.values())).causal_result
        var_names = first_res.var_names
        target_var = "d_CS_i"
        j = var_names.index(target_var)
        tau_max = first_res.tau_max
        alpha = first_res.pc_alpha

        rows = []
        for i, driver in enumerate(var_names):
            for tau in range(tau_max + 1):
                if driver == target_var and tau == 0:
                    continue
                row = {"Driver": driver, "Lag": -tau}
                any_sig = False
                for key, prun in runs.items():
                    res = prun.causal_result
                    val = float(res.val_matrix[i, j, tau])
                    p = float(res.p_matrix[i, j, tau])
                    sig = (p <= alpha) and (res.graph[i, j, tau] != "")
                    any_sig = any_sig or sig
                    row[f"{key}_MCI"] = round(val, 4)
                    row[f"{key}_p"] = round(p, 6)
                    row[f"{key}_sig"] = sig
                row["Any_Sig"] = any_sig
                rows.append(row)

        df_out = pd.DataFrame(rows)
        return df_out[df_out["Any_Sig"]].drop(columns=["Any_Sig"]).reset_index(drop=True)

    @staticmethod
    def print_report(runs: Dict[str, PanelDiscoveryResult], summary_df: pd.DataFrame):
        print("\n" + "=" * 105)
        print(" PHASE 3: MICRO PANEL CAUSAL DISCOVERY (d_CS_i | Maturity-Matched Corporate Bond Spreads)")
        print("=" * 105)
        for key, prun in runs.items():
            print(f"  {key:<16} | {prun.label:<30} | Bonds={prun.n_bonds:<3} | Obs={prun.n_total_obs:,}")
        print("-" * 105)

        disp = pd.DataFrame({"Driver": summary_df["Driver"], "Lag": summary_df["Lag"]})
        for key in runs:
            stars = summary_df[f"{key}_p"].apply(
                lambda p: "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
            )
            cell = summary_df[f"{key}_MCI"].map("{:+.4f}".format) + stars
            disp[key] = np.where(summary_df[f"{key}_sig"], cell, cell + " (ns)")

        print(disp.to_string(index=False))
        print("=" * 105 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    panel_disc = PanelCausalDiscovery(tau_max=3, pc_alpha=0.01, min_bond_obs=150)
    df_panel = panel_disc.load_and_prepare_panel()
    runs = panel_disc.run_regime_and_merton_splits(df_panel)
    summary_df = panel_disc.build_summary_table(runs)
    panel_disc.print_report(runs, summary_df)
