# Literature Review: Causal Inference & Time-Series Forecasting

This directory synthesizes foundational papers across causal inference, time-series transformers, and fixed-income finance. 

**The Core Hypothesis:** Standard deep learning models overfit to false historical correlations during macroeconomic regime shifts. By using a causal discovery algorithm to filter macro features before passing them to a multivariate transformer, we can isolate structural economic drivers and build a regime-robust forecasting pipeline.

---

### 1. The Domain Baseline
*These papers define our target variables and highlight the flaw in current quantitative models.*

*   **The Determinants of Credit Spread Changes (2001)**
    Establishes the theoretical macro drivers of credit spreads (e.g., risk-free rates, yield curve slope, VIX). This gives us the exact list of macroeconomic variables we need to download from the FRED database to use as our model's inputs.
*   **A Novel Methodology in Credit Spread Prediction Based on Machine Learning (2024)**
    The current industry baseline. Proves ML can accurately map macro features to spreads, but relies only on surface-level historical correlations, meaning the model will likely break the moment the broader economic environment changes.

### 2. Causal Discovery & Regime Shifts
*These papers justify why correlation fails and how causal mapping fixes the blind spot.*

*   **Causal Factor Investing (2019)**
    Argues that financial backtesting is inherently confounded. Without a structural causal graph, models merely memorize historical coincidences driven by unobserved variables.
*   **Regime-Aware Causal Bayesian Forecasting (2026)**
    Proves that causal structures in time series change depending on the active regime. Validates the need to treat historical macro data as a sequence of shifting environments.
*   **Inferring Causation from Time Series in Earth System Sciences (2019)**
    Introduces the PCMCI algorithm. Uses Momentary Conditional Independence (MCI) to filter out the repetitive noise in economic data, allowing us to isolate the exact macroeconomic shocks that cause spreads to move.

### 3. The Deep Learning Backbone
*These papers define the neural architecture that will ingest our causally filtered data.*

*   **Non-stationary Transformers: Exploring the Blind Spot (2022)**
    Proves that standard transformers mathematically fail when economic conditions change rapidly, which is why we must causally filter our data before feeding it into the model.
*   **iTransformer: Inverted Transformers Are Effective for Time Series (2024)**
    Introduces cross-variate attention. By calculating attention *across variables* rather than across time steps, it explicitly captures how distinct macroeconomic shocks interact to drive credit spreads.
*   **Predicting Liquidity-Aware Bond Yields using Causal GANs (2025)**
    Validates our direction. Demonstrates that the active frontier of quantitative finance is currently converging on the integration of causal models and deep learning.

---

**Proposed Pipeline Architecture:**  
`Raw Macro Data (FRED, WRDS) ➔ Causal Pruning ➔ Robust Features ➔ Cross-Variate Attention Engine (iTransformer) ➔ Regime-Aware Forecast`
