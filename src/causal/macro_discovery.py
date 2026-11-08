"""
Runs PCMCI+ on the stationarized 2000-2026 macro series across distinct monetary
regimes, then classifies every candidate driver of the credit spread change as:

    STABLE_CORE      - significant in every comparison regime with the same sign
    SIGN_FLIP        - significant in every comparison regime but the sign changes
    REGIME_SPECIFIC  - significant in some comparison regimes but not others
    INSIGNIFICANT    - not significant in any comparison regime

The full-sample run is reported as a baseline but is not used for classification,
since it pools the very regimes we are trying to compare.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .engine import CausalEngine, CausalResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_stationary_2000_2026.csv"

# d_DEF_PREM is excluded by default: DEF_PREM = CS_BBB - CS_AAA contains the target
# by construction, so its "causal" link to d_CS_BBB would be partly an accounting identity.
DEFAULT_MACRO_VARS = ["d_CS_BBB", "d_DGS10", "d_T10Y2Y", "ret_SPY", "d_VIX"]


@dataclass(frozen=True)
class MacroRegime:
    """A named, contiguous date window representing a macro/monetary regime."""
    name: str
    start_date: str
    end_date: str
    description: str = ""
    is_baseline: bool = False  # Baseline regimes are reported but excluded from classification


DEFAULT_REGIMES: List[MacroRegime] = [
    MacroRegime(
        name="FULL",
        start_date="2000-01-01",
        end_date="2026-10-01",
        description="Full 26-year history (pooled baseline)",
        is_baseline=True,
    ),
    MacroRegime(
        name="ZIRP",
        start_date="2010-01-01",
        end_date="2019-12-31",
        description="ZIRP / QE expansion: near-zero policy rate, low rate volatility",
    ),
    MacroRegime(
        name="TIGHTENING",
        start_date="2022-01-01",
        end_date="2026-10-01",
        description="Inflation shock & tightening: +525bps hikes, prolonged curve inversion",
    ),
]


@dataclass
class RegimeRun:
    """Result of a single regime's PCMCI+ run."""
    regime: MacroRegime
    n_obs: int
    result: CausalResult


@dataclass
class MacroCausalDiscovery:
    """Orchestrates regime-partitioned PCMCI+ discovery and invariance classification."""
    var_names: List[str] = field(default_factory=lambda: list(DEFAULT_MACRO_VARS))
    target_var: str = "d_CS_BBB"
    tau_max: int = 5
    pc_alpha: float = 0.01
    regimes: List[MacroRegime] = field(default_factory=lambda: list(DEFAULT_REGIMES))
    engine: CausalEngine = field(default_factory=CausalEngine)
    runs: Dict[str, RegimeRun] = field(default_factory=dict, init=False)

    @staticmethod
    def load_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df.sort_index()

    def run_all_regimes(self, df: pd.DataFrame) -> Dict[str, RegimeRun]:
        """Runs PCMCI+ with identical hyperparameters on each regime slice."""
        missing = [v for v in self.var_names if v not in df.columns]
        if missing:
            raise ValueError(f"Variables missing from data: {missing}")

        for regime in self.regimes:
            df_slice = df.loc[regime.start_date:regime.end_date, self.var_names].dropna()
            min_obs = 20 * (self.tau_max + 1) * len(self.var_names)
            if len(df_slice) < min_obs:
                logger.warning(
                    f"[{regime.name}] only {len(df_slice)} obs (< {min_obs}); results may be unreliable."
                )

            logger.info(f"[{regime.name}] {regime.start_date} -> {regime.end_date}, N={len(df_slice)}")
            result = self.engine.run(
                df_slice,
                var_names=self.var_names,
                target_var=self.target_var,
                tau_max=self.tau_max,
                pc_alpha=self.pc_alpha,
            )
            self.runs[regime.name] = RegimeRun(regime=regime, n_obs=len(df_slice), result=result)

        return self.runs

    def _edge_stats(self, run: RegimeRun, driver: str, tau: int) -> Dict:
        """MCI value, p-value, graph edge and significance for one (driver, tau) -> target edge."""
        res = run.result
        i = res.var_names.index(driver)
        j = res.var_names.index(self.target_var)
        val = float(res.val_matrix[i, j, tau])
        p = float(res.p_matrix[i, j, tau])
        edge = str(res.graph[i, j, tau])
        sig = (p <= self.pc_alpha) and edge != ""
        return {"val": val, "p": p, "edge": edge, "sig": sig}

    def build_comparison_matrix(self) -> pd.DataFrame:
        """
        One row per candidate edge (driver, lag) -> target, with MCI effect, p-value and
        significance in each regime, plus a stability classification.
        """
        if not self.runs:
            raise RuntimeError("Call run_all_regimes() first.")

        compare_names = [n for n, r in self.runs.items() if not r.regime.is_baseline]
        rows = []

        for driver in self.var_names:
            for tau in range(self.tau_max + 1):
                if driver == self.target_var and tau == 0:
                    continue  # no self-loop at lag 0

                row = {"Driver": driver, "Lag": -tau}
                for name, run in self.runs.items():
                    s = self._edge_stats(run, driver, tau)
                    row[f"{name}_MCI"] = round(s["val"], 4)
                    row[f"{name}_p"] = round(s["p"], 6)
                    row[f"{name}_sig"] = s["sig"]

                sig_flags = [row[f"{n}_sig"] for n in compare_names]
                signs = [np.sign(row[f"{n}_MCI"]) for n in compare_names if row[f"{n}_sig"]]

                if all(sig_flags):
                    status = "STABLE_CORE" if len(set(signs)) == 1 else "SIGN_FLIP"
                elif any(sig_flags):
                    status = "REGIME_SPECIFIC"
                else:
                    status = "INSIGNIFICANT"
                row["Status"] = status
                rows.append(row)

        df = pd.DataFrame(rows)
        order = {"STABLE_CORE": 0, "SIGN_FLIP": 1, "REGIME_SPECIFIC": 2, "INSIGNIFICANT": 3}
        df["_o"] = df["Status"].map(order)
        df = df.sort_values(["_o", "Lag", "Driver"], ascending=[True, False, True]).drop(columns="_o")
        return df.reset_index(drop=True)

    def print_summary(self, comparison: Optional[pd.DataFrame] = None, show_insignificant: bool = False):
        comparison = comparison if comparison is not None else self.build_comparison_matrix()

        print("\n" + "=" * 90)
        print(f" CROSS-REGIME CAUSAL DRIVERS OF {self.target_var}  (tau_max={self.tau_max}, alpha={self.pc_alpha})")
        print("=" * 90)
        for name, run in self.runs.items():
            tag = " [baseline]" if run.regime.is_baseline else ""
            print(f"  {name:<11} {run.regime.start_date} -> {run.regime.end_date}  N={run.n_obs:<5}{tag}")
        print("-" * 90)

        view = comparison if show_insignificant else comparison[comparison["Status"] != "INSIGNIFICANT"]
        display = pd.DataFrame({"Driver": view["Driver"], "Lag": view["Lag"]})
        for name in self.runs:
            stars = view[f"{name}_p"].apply(
                lambda p: "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
            )
            cell = view[f"{name}_MCI"].map("{:+.3f}".format) + stars
            display[name] = np.where(view[f"{name}_sig"], cell, cell + " (ns)")
        display["Status"] = view["Status"]
        print(display.to_string(index=False))
        print("-" * 90)
        print("  *** p<=0.001  ** p<=0.01  * p<=0.05   (ns) = not retained by PCMCI+ at alpha")
        print("  Classification uses non-baseline regimes only.")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    discovery = MacroCausalDiscovery(tau_max=5, pc_alpha=0.01)
    df = discovery.load_data()
    discovery.run_all_regimes(df)
    comparison = discovery.build_comparison_matrix()
    discovery.print_summary(comparison)

