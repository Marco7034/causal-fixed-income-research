# Causal-Gated Credit Spread Forecaster
# GOAL Lab — Queen's University
"""
src package for the Causal Fixed-Income Research pipeline.

Modules:
    data_pipeline      – FRED macro & Yahoo Finance data ingestion, alignment, and feature engineering
    wrds_pipeline      – WRDS micro-data extraction (TRACE, CRSP, Compustat, FISD)
    build_master_panel – Cross-sectional panel construction merging macro and micro streams
    causal_discovery   – PCMCI+ causal gate for feature selection
"""

