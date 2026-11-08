"""
Wraps Tigramite's PCMCI+ algorithm with ParCorr conditional independence testing,
structural link prior enforcement, and standardized causal parent extraction.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from .priors import EXPECTED_SIGNS, build_link_assumptions

logger = logging.getLogger(__name__)


@dataclass
class CausalResult:
    """Encapsulates the output of a PCMCI+ causal discovery run."""
    var_names: List[str]
    tau_max: int
    pc_alpha: float
    val_matrix: np.ndarray        # Shape: (N, N, tau_max + 1)
    p_matrix: np.ndarray          # Shape: (N, N, tau_max + 1)
    graph: np.ndarray             # Shape: (N, N, tau_max + 1) string array
    parents_df: pd.DataFrame      # Formatted causal parents of the target variable

    def get_adjacency_matrix(self, alpha: Optional[float] = None) -> np.ndarray:
        """
        Returns a binary adjacency matrix across all variables and lags.
        Shape: (N, N, tau_max + 1), where entry (i, j, tau) = 1 if edge is significant.
        """
        crit_alpha = alpha or self.pc_alpha
        sig_mask = (self.p_matrix <= crit_alpha) & (self.graph != "")
        return sig_mask.astype(np.int32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes core metadata and parent tables for artifact export."""
        return {
            "var_names": self.var_names,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "parents": self.parents_df.to_dict(orient="records"),
        }


class CausalEngine:
    """
    Stateless executor for PCMCI+ causal discovery on stationary financial time series.
    """

    def __init__(
        self,
        cond_ind_test: Optional[Any] = None,
        significance: str = "analytic",
        verbosity: int = 0
    ):
        """
        Parameters:
        cond_ind_test : Optional[Any]
            Tigramite conditional independence test instance. Defaults to ParCorr(significance=significance).
        significance : str, default "analytic"
            Significance method for ParCorr ('analytic', 'shuffle_test', or 'bootstrap').
        verbosity : int, default 0
            Tigramite verbosity level (0 = silent, 1 = intermediate logs).
        """
        self.cond_ind_test = cond_ind_test or ParCorr(significance=significance)
        self.verbosity = verbosity

    @staticmethod
    def df_to_tigramite(
        df: pd.DataFrame,
        var_names: Optional[List[str]] = None
    ) -> Tuple[pp.DataFrame, List[str]]:
        """
        Converts a Pandas DataFrame into Tigramite's internal DataFrame format.
        """
        cols = var_names or list(df.columns)
        clean_df = df[cols].dropna()

        data_matrix = clean_df.values.astype(np.float64)
        datatime = np.arange(len(clean_df), dtype=np.float64)

        tig_df = pp.DataFrame(
            data=data_matrix,
            datatime=datatime,
            var_names=cols
        )
        return tig_df, cols

    @staticmethod
    def extract_target_parents(
        val_matrix: np.ndarray,
        p_matrix: np.ndarray,
        graph: np.ndarray,
        var_names: List[str],
        target_var: str,
        alpha_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Extracts and formats all statistically verified causal parents of `target_var`.

        Returns:
        pd.DataFrame
            Columns: Driver, Lag (tau), Relation, Graph_Edge, MCI_Effect,
                     p_value, Sign, Expected_Sign, Theory_Match.
        """
        if target_var not in var_names:
            raise ValueError(f"target_var '{target_var}' not found in var_names: {var_names}")

        target_idx = var_names.index(target_var)
        n_vars, _, max_lag_plus_1 = val_matrix.shape

        rows = []
        for i in range(n_vars):
            source_var = var_names[i]
            for tau in range(max_lag_plus_1):
                p_val = p_matrix[i, target_idx, tau]
                mci_val = val_matrix[i, target_idx, tau]
                edge_str = graph[i, target_idx, tau]

                # Significant incoming edge into target_var
                if p_val <= alpha_level and edge_str != "":
                    actual_sign = "+" if mci_val > 0 else "-"
                    exp_sign = EXPECTED_SIGNS.get(source_var, "?")
                    theory_match = (actual_sign == exp_sign) if exp_sign != "?" else None

                    rows.append({
                        "Driver": source_var,
                        "Lag (tau)": -tau,
                        "Relation": "Contemp (t)" if tau == 0 else f"Lagged (t-{tau})",
                        "Graph_Edge": edge_str,
                        "MCI_Effect": round(float(mci_val), 4),
                        "p_value": round(float(p_val), 6),
                        "Sign": actual_sign,
                        "Expected_Sign": exp_sign,
                        "Theory_Match": theory_match
                    })

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            # Sort by absolute effect size descending
            df_out["_abs_eff"] = df_out["MCI_Effect"].abs()
            df_out = df_out.sort_values(by=["Lag (tau)", "_abs_eff"], ascending=[False, False])
            df_out = df_out.drop(columns=["_abs_eff"]).reset_index(drop=True)

        return df_out

    def run(
        self,
        df: pd.DataFrame,
        var_names: List[str],
        target_var: str = "d_CS_BBB",
        tau_min: int = 0,
        tau_max: int = 5,
        pc_alpha: float = 0.01,
        link_assumptions: Optional[Dict] = None,
        apply_priors: bool = True,
    ) -> CausalResult:
        """
        Executes the PCMCI+ algorithm on the input DataFrame.

        Parameters:
        df : pd.DataFrame
            Stationarized time series data.
        var_names : List[str]
            Subset of columns to include in discovery.
        target_var : str, default "d_CS_BBB"
            Target spread change variable.
        tau_min : int, default 0
            Minimum lag to evaluate (0 includes contemporaneous tests).
        tau_max : int, default 5
            Maximum lag to evaluate.
        pc_alpha : float, default 0.01
            Significance threshold for conditional independence.
        link_assumptions : Optional[Dict]
            Custom link assumptions. If None and apply_priors is True,
            `build_link_assumptions` is automatically constructed.
        apply_priors : bool, default True
            Whether to enforce macroeconomic contemporaneous primacy constraints.

        Returns:
        CausalResult
        """
        tig_df, cols = self.df_to_tigramite(df, var_names)

        # Build link assumptions if requested
        assumptions = link_assumptions
        if assumptions is None and apply_priors:
            assumptions = build_link_assumptions(cols, target_var=target_var, tau_max=tau_max)

        pcmci = PCMCI(
            dataframe=tig_df,
            cond_ind_test=self.cond_ind_test,
            verbosity=self.verbosity
        )

        logger.info(
            f"Running PCMCI+ on N={len(df)} obs, vars={cols}, "
            f"tau_min={tau_min}, tau_max={tau_max}, alpha={pc_alpha}"
        )

        results = pcmci.run_pcmciplus(
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            link_assumptions=assumptions
        )

        parents_df = self.extract_target_parents(
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            graph=results["graph"],
            var_names=cols,
            target_var=target_var,
            alpha_level=pc_alpha
        )

        return CausalResult(
            var_names=cols,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            graph=results["graph"],
            parents_df=parents_df
        )

