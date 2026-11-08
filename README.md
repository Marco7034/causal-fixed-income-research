# The Causal Determinants of Credit Spread Changes and Regime-Robust Forecasting

> An empirical framework combining structural credit risk theory with time-series causal discovery to isolate regime-invariant drivers of corporate bond spreads and achieve robust out-of-sample forecasting.

---

## Abstract & Core Hypothesis

Quantitative credit spread models—from classical Vector Autoregressions to modern deep learning architectures—rely heavily on historical correlations. In fixed-income markets, these correlations are confounded by common monetary policy impulses and frequently decouple or invert when macroeconomic regimes shift (e.g., from zero-rate expansion to aggressive monetary tightening). Unconstrained models overfit to these transient correlations, leading to severe performance collapse out-of-sample.

We propose an upstream Causal Gate grounded in structural credit risk theory (Merton, 1974; Collin-Dufresne, Goldstein, and Martin, 2001). By leveraging Runge's PCMCI+ algorithm to evaluate momentary conditional independence across shifting monetary regimes, we prune spurious and regime-fragile links. Our core hypothesis is that restricting the predictive feature space to theoretically and empirically verified causal parents prevents overfitting to policy-specific correlations and produces superior, regime-invariant out-of-sample credit spread forecasts.

---

## Data & Methodology

The empirical pipeline is structured into two complementary layers:

* Macro Layer (2000–2026, 6,684 Daily Observations):
  * *Target:* Moody's Baa Corporate Bond Spread over 10Y Treasury (`CS_BBB`).
  * *Drivers:* 10-Year Treasury Yield (`DGS10`), 10Y–2Y Yield Curve Slope (`T10Y2Y`), S&P 500 Equity Returns (`SPY`), and CBOE Volatility Index (`VIX`).
  * *Regimes:* Full 26-year baseline, ZIRP/QE expansion (2010–2019), and post-COVID inflation tightening (2022–2026).
* Micro Panel Layer (2019–2023, 161 Liquid Bonds, 193,508 Observations):
  * *Target:* Individual bond credit spreads (`CS_i`), dynamically maturity-matched against linearly interpolated benchmark Treasuries (eliminating term-structure slope distortion).
  * *Drivers:* Firm-specific daily equity returns (CRSP), quarterly book leverage (Compustat, strictly lagged to SEC `rdq` filing dates), and secondary trading volume (TRACE).
* Causal Discovery & Forecasting Engine:
  * *Causal Gate (PCMCI+):* Multi-entity conditional independence testing with structural macroeconomic contemporaneous primacy constraints ($\tau = 0$) and unconstrained lagged feedback ($\tau \ge 1$).
  * *Out-of-Sample Evaluation:* Models trained strictly on pre-tightening data and tested out-of-sample on the 2022+ tightening shock using strictly lagged predictors ($\tau \ge 1$).

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── figures/
│   ├── credit_spread_overview.png          # 26-year macro regime time series
│   └── baseline_forecast_comparison.png    # Out-of-sample benchmark forecast comparison
├── data/
│   ├── raw/
│   │   ├── data_levels_2000_2026.csv       # Raw macro series (FRED & Yahoo Finance)
│   │   ├── wrds_trace_yields.csv           # Secondary TRACE bond yields
│   │   ├── wrds_crsp_equity.csv            # CRSP firm-level equity returns
│   │   ├── wrds_compustat_leverage.csv     # Compustat balance sheet leverage
│   │   └── wrds_bond_chars.csv             # Enriched bond maturity and coupon metadata
│   └── processed/
│       ├── data_stationary_2000_2026.csv   # Stationarized macro shocks
│       └── MASTER_MODEL_PANEL.csv          # Aligned, maturity-matched bond panel
└── src/
    ├── build_master_panel.py               # Panel construction & Treasury interpolation
    ├── data_pipeline.py                    # Macro FRED/YFinance data pipeline
    ├── wrds_pipeline.py                    # WRDS SQL extraction pipelines
    ├── inspect_data.py                     # Exploratory analysis & regime plotting
    ├── causal/
    │   ├── priors.py                       # Merton expected signs & link constraints
    │   ├── engine.py                       # PCMCI+ wrapper & causal parent extractor
    │   ├── macro_discovery.py              # Macro cross-regime causal discovery
    │   └── panel_discovery.py              # Multi-entity micro panel causal discovery
    └── models/
        └── baseline.py                     # Out-of-sample forecast benchmark evaluation
```

---

## Quickstart & Reproduction

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Execution
```bash
# Step 1: Build the maturity-matched master panel from raw data
python3 src/build_master_panel.py

# Step 2: Run macro cross-regime causal discovery (ZIRP vs. Tightening)
python3 -m src.causal.macro_discovery

# Step 3: Run multi-entity micro panel causal discovery
python3 -m src.causal.panel_discovery

# Step 4: Run out-of-sample baseline vs. causal-gated forecast evaluation
python3 -m src.models.baseline
```
