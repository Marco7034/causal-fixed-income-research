import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the Fixed-Income Causal Discovery Data Pipeline."""
    start_date: str = "2000-01-01"
    end_date: str = "2026-10-01"
    frequency: str = "B"  # "B" (Business Daily) or "W-FRI" (Weekly Friday close)
    target_col: str = "CS_BBB"
    max_ffill_days: int = 5  # Maximum consecutive days to forward-fill macro/market holidays
    
    # Primary FRED series (Full 2000-2026 Daily History, Zero Vendor Truncation)
    fred_series: Dict[str, str] = field(default_factory=lambda: {
        # Primary Target: Moody's Baa (= BBB) Corporate Bond Spread over 10Y Treasury
        "CS_BBB": "BAA10Y",            # Full daily history
        # Collin-Dufresne (2001) Structural Determinants
        "DGS10": "DGS10",              # 10-Year Treasury Constant Maturity Rate (Spot Rate)
        "T10Y2Y": "T10Y2Y",            # 10-Year Minus 2-Year Treasury Spread (Yield Curve Slope)
        # Systemic Liquidity / Default Quality Spread Proxy
        "CS_AAA": "AAA10Y",            # Moody's Aaa (= AAA) Corporate Bond Spread over 10Y Treasury
        # Monetary & Liquidity Regime Conditioners
        "DFF": "DFF",                  # Effective Federal Funds Rate (Daily, full 2000-2026 history)
        "NFCI": "NFCI",                # Chicago Fed National Financial Conditions Index (Weekly)
    })
    
    # Yahoo Finance Tickers (Equity Returns & Implied Volatility)
    yf_tickers: Dict[str, str] = field(default_factory=lambda: {
        "SPY": "SPY",                  # S&P 500 ETF
        "VIX": "^VIX",                 # CBOE Volatility Index
    })


class MacroFixedIncomePipeline:
    def __init__(self, config: PipelineConfig, fred_api_key: Optional[str] = None):
        self.cfg = config
        self.api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key not found. Pass `fred_api_key` or set `export FRED_API_KEY='...'` in zsh."
            )
        self.fred = Fred(api_key=self.api_key)

    def fetch_fred_data(self) -> pd.DataFrame:
        """
        Fetches macroeconomic and fixed-income series from FRED.
        Uses direct FRED graph CSV endpoint for ICE BofA (BAML*) series to bypass
        the 3-year API redistributor truncation, and fredapi for standard series.
        """
        logger.info("Fetching FRED series")
        series_dict = {}
        start_dt = pd.to_datetime(self.cfg.start_date)
        end_dt = pd.to_datetime(self.cfg.end_date)

        for name, code in self.cfg.fred_series.items():
            try:
                if code.startswith("BAML"):
                    # Direct FRED CSV endpoint returns full history (1997-present)
                    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
                    df_csv = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
                    s = df_csv[code].loc[start_dt:end_dt].astype(float)
                else:
                    s = self.fred.get_series(
                        code,
                        observation_start=self.cfg.start_date,
                        observation_end=self.cfg.end_date
                    )
                s.name = name
                series_dict[name] = s
                logger.info(f"  [FRED] {name} ({code}): {len(s.dropna())} valid observations.")
            except Exception as e:
                logger.warning(f"  [FRED] Failed to fetch {name} ({code}): {e}")

        df_fred = pd.DataFrame(series_dict)
        df_fred.index = pd.to_datetime(df_fred.index).tz_localize(None)
        return df_fred

    def fetch_yf_data(self) -> pd.DataFrame:
        """Fetches market equity and implied volatility series from Yahoo Finance."""
        logger.info("Fetching Yahoo Finance series")
        series_dict = {}
        for name, ticker in self.cfg.yf_tickers.items():
            df_t = yf.download(
                ticker,
                start=self.cfg.start_date,
                end=self.cfg.end_date,
                progress=False,
                auto_adjust=True
            )
            if df_t.empty:
                raise RuntimeError(f"Yahoo Finance returned empty dataframe for {ticker}")
            
            # Extract Close (already adjusted when auto_adjust=True)
            close_s = df_t["Close"].squeeze()
            close_s.name = name
            close_s.index = pd.to_datetime(close_s.index).tz_localize(None)
            series_dict[name] = close_s
            logger.info(f"  [YFinance] {name} ({ticker}): {len(close_s.dropna())} valid observations.")

        return pd.DataFrame(series_dict)

    def align_and_clean(self, df_fred: pd.DataFrame, df_yf: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns asynchronous trading calendars (SIFMA vs. NYSE) without lookahead bias.
        - Never interpolates.
        - Forward-fills exogenous variables up to `max_ffill_days` to carry last known market state.
        - Anchors on days where the target variable was actively observed.
        """
        logger.info("Aligning calendars and handling missing observations (strict causal ffill)...")
        df_raw = pd.concat([df_fred, df_yf], axis=1).sort_index()

        # Generate standard business day grid
        bday_index = pd.date_range(
            start=df_raw.index.min(),
            end=df_raw.index.max(),
            freq="B"
        )
        df_aligned = df_raw.reindex(bday_index)

        # Separate target from features so we only keep days where the target was actually priced,
        # while allowing features from t-1 (e.g., NYSE open when bond market closed or vice versa) to ffill.
        target_valid_mask = df_aligned[self.cfg.target_col].notna()
        
        feature_cols = [c for c in df_aligned.columns if c != self.cfg.target_col]
        df_aligned[feature_cols] = df_aligned[feature_cols].ffill(limit=self.cfg.max_ffill_days)

        # Keep only dates with valid target observations
        df_clean = df_aligned.loc[target_valid_mask].copy()

        # Optional resampling
        if self.cfg.frequency != "B":
            logger.info(f"Resampling levels to frequency: '{self.cfg.frequency}' (last observed value)...")
            df_clean = df_clean.resample(self.cfg.frequency).last()

        # Drop any leading rows where slower-starting series (or initial NaNs) are missing
        core_cols = [self.cfg.target_col, "DGS10", "T10Y2Y", "SPY", "VIX"]
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=core_cols)
        logger.info(f"Aligned dataset: {len(df_clean)} rows (dropped {initial_len - len(df_clean)} incomplete rows).")
        return df_clean

    def engineer_structural_features(self, df_levels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constructs:
        1. `df_levels`: Raw levels + derived structural levels (for regime detection & iTransformer context).
        2. `df_stationary`: First-differenced / log-return series following Collin-Dufresne et al. (2001)
           for PCMCI+ causal discovery.
        """
        logger.info("Engineering Collin-Dufresne (2001) structural features and stationarized shocks")
        df_lvl = df_levels.copy()

        # Derived structural level: Default/Quality Spread (BBB OAS minus AAA OAS)
        if "CS_BBB" in df_lvl.columns and "CS_AAA" in df_lvl.columns:
            df_lvl["DEF_PREM"] = df_lvl["CS_BBB"] - df_lvl["CS_AAA"]

        df_stat = pd.DataFrame(index=df_lvl.index)

        # Target: Change in BBB Credit Spread (\Delta CS_t)
        df_stat["d_CS_BBB"] = df_lvl["CS_BBB"].diff()

        # Spot Rate Shock: Change in 10Y Treasury Yield (\Delta r^{10Y}_t)
        df_stat["d_DGS10"] = df_lvl["DGS10"].diff()

        # Convexity / Non-linear Spot Rate term from Collin-Dufresne Table VIII
        df_stat["d_DGS10_sq"] = (df_lvl["DGS10"].diff()) ** 2

        # Yield Curve Slope Shock: Change in 10Y-2Y Spread
        df_stat["d_T10Y2Y"] = df_lvl["T10Y2Y"].diff()

        # Equity Market Return: Log Return of S&P 500
        df_stat["ret_SPY"] = np.log(df_lvl["SPY"] / df_lvl["SPY"].shift(1))

        # Implied Volatility Shock: Change in VIX
        df_stat["d_VIX"] = df_lvl["VIX"].diff()

        # Default/Liquidity Premium Shock
        if "DEF_PREM" in df_lvl.columns:
            df_stat["d_DEF_PREM"] = df_lvl["DEF_PREM"].diff()

        # Drop the initial NaN created by first-differencing
        df_stat = df_stat.dropna()
        df_lvl = df_lvl.loc[df_stat.index]

        return df_lvl, df_stat

    @staticmethod
    def run_stationarity_diagnostics(df: pd.DataFrame, label: str = "Dataset") -> pd.DataFrame:
        """Runs Augmented Dickey-Fuller (ADF) unit-root tests to verify stationarity for PCMCI+."""
        logger.info(f"Running ADF Stationarity Diagnostics on {label}...")
        results = []
        for col in df.columns:
            series = df[col].dropna()
            adf_stat, p_val, used_lag, nobs, crit_vals, _ = adfuller(
                series, autolag="AIC", result_object=False
            )
            results.append({
                "Variable": col,
                "ADF_Stat": round(adf_stat, 4),
                "p_value": round(p_val, 6),
                "1%_Crit": round(crit_vals["1%"], 3),
                "Stationary_at_1%": adf_stat < crit_vals["1%"],
                "Lag_1_Autocorr": round(series.autocorr(lag=1), 4)
            })
        res_df = pd.DataFrame(results)
        print(f"\n--- ADF Stationarity & Autocorrelation Report ({label}) ---")
        print(res_df.to_string(index=False))
        print("-" * 65 + "\n")
        return res_df

    @staticmethod
    def to_tigramite_format(df_stat: pd.DataFrame, cols: Optional[List[str]] = None):
        """
        Converts the stationarized pandas DataFrame into a `tigramite.data_processing.DataFrame`
        ready for PCMCI+ / LPCMCI / RPCMCI execution.
        """
        from tigramite import data_processing as pp

        selected_cols = cols or [c for c in df_stat.columns if c != "d_DGS10_sq"]
        data_matrix = df_stat[selected_cols].values.astype(np.float64)
        datatime = np.arange(len(df_stat), dtype=np.float64)

        tig_df = pp.DataFrame(
            data=data_matrix,
            datatime=datatime,
            var_names=selected_cols
        )
        logger.info(f"Created Tigramite DataFrame with shape {data_matrix.shape} and variables: {selected_cols}")
        return tig_df, selected_cols


if __name__ == "__main__":
    # Initialize Configuration
    config = PipelineConfig(
        start_date="2000-01-01",
        end_date="2026-10-01",
        frequency="B"
    )

    pipeline = MacroFixedIncomePipeline(config)
    
    # Fetch & Align
    df_fred = pipeline.fetch_fred_data()
    df_yf = pipeline.fetch_yf_data()
    df_aligned = pipeline.align_and_clean(df_fred, df_yf)

    # Engineer Levels & Stationarized Shocks
    df_levels, df_stationary = pipeline.engineer_structural_features(df_aligned)

    # Verify Stationarity & Autocorrelation Structure
    pipeline.run_stationarity_diagnostics(df_levels[["CS_BBB", "DGS10", "T10Y2Y", "SPY", "VIX"]], label="Raw Levels")
    pipeline.run_stationarity_diagnostics(df_stationary, label="First-Differenced Shocks")

    # Prepare Tigramite Object for Step 2
    causal_vars = ["d_CS_BBB", "d_DGS10", "d_T10Y2Y", "ret_SPY", "d_VIX", "d_DEF_PREM"]
    tigramite_df, var_names = pipeline.to_tigramite_format(df_stationary, cols=causal_vars)