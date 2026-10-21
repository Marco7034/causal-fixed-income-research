from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Maximum business days to forward-fill quarterly Compustat leverage.
# ~90 bdays ≈ 1 quarter. Prevents unbounded carry-forward if a firm misses filing.
COMPUSTAT_FFILL_LIMIT = 90


def build_master_panel():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FRED macro data and WRDS micro data...")
    # Load FRED
    df_fred = pd.read_csv(RAW_DIR / "data_levels_2000_2026.csv", index_col=0, parse_dates=True)
    df_fred.index.name = "date"
    
    # Load WRDS
    df_trace = pd.read_csv(RAW_DIR / "wrds_trace_yields.csv", parse_dates=["date"])
    df_crsp = pd.read_csv(RAW_DIR / "wrds_crsp_equity.csv", parse_dates=["date"])
    df_comp = pd.read_csv(RAW_DIR / "wrds_compustat_leverage.csv", parse_dates=["date"])
    
    # Load FISD bond characteristics
    fisd_path = RAW_DIR / "wrds_bond_chars.csv"
    df_fisd = None
    if fisd_path.exists():
        df_fisd = pd.read_csv(fisd_path, parse_dates=["maturity_date"])
        logger.info(f"Loaded FISD bond characteristics for {len(df_fisd)} bonds.")
    else:
        logger.warning(
            f"FISD file not found at {fisd_path}. "
            "Bond-level features (coupon, TTM, seniority) will be skipped. "
            "Run wrds_pipeline.py to generate it."
        )

    # Compustat filing-date correction & bounded forward-fill
    logger.info("1. Forward-filling quarterly Compustat leverage to daily frequency...")
    logger.info(f"   Filing-date lag: using 'date' column (rdq or fiscal_date + 45d from wrds_pipeline)")
    logger.info(f"   Forward-fill limit: {COMPUSTAT_FFILL_LIMIT} business days (~1 quarter)")
    
    # The 'date' column in df_comp is already the filing-availability date
    # (set by wrds_pipeline.py using rdq or datadate + 45 calendar days).
    # We resample to business-day frequency per issuer with a bounded ffill.
    comp_daily = (
        df_comp
        .set_index('date')
        .groupby('issuer_cusip')[['leverage']]
        .resample('B')
        .ffill(limit=COMPUSTAT_FFILL_LIMIT)
        .reset_index()
    )

    # Merge TRACE + CRSP equity + Compustat fundamentals
    logger.info("2. Merging TRACE yields with CRSP equity and Compustat fundamentals...")
    # Base panel is the TRACE bond transaction days
    panel = pd.merge(
        df_trace, 
        df_crsp[['issuer_cusip', 'date', 'daily_return', 'equity_volume', 'market_cap']], 
        on=['issuer_cusip', 'date'], how='inner'
    )
    
    # Merge Compustat (leverage now correctly lagged to filing date)
    panel = pd.merge(
        panel, comp_daily[['issuer_cusip', 'date', 'leverage']], 
        on=['issuer_cusip', 'date'], how='left'
    )
    
    # Within-panel ffill for residual gaps
    panel = panel.sort_values(['cusip', 'date'])
    panel['leverage'] = panel.groupby('issuer_cusip')['leverage'].ffill(limit=COMPUSTAT_FFILL_LIMIT)

    # Merge FISD bond-level features (coupon, TTM, seniority)
    if df_fisd is not None:
        logger.info("3a. Merging FISD bond characteristics (coupon, maturity, seniority)...")
        
        # Merge on the 9-digit bond CUSIP
        panel = pd.merge(
            panel,
            df_fisd[['cusip', 'coupon', 'maturity_date', 'seniority']],
            on='cusip', how='left'
        )
        
        # Compute Time-to-Maturity in years
        panel['ttm_years'] = (
            (panel['maturity_date'] - panel['date']).dt.days / 365.25
        )
        
        # Create seniority indicator
        panel['is_senior'] = panel['seniority'].isin(['Senior', 'Senior Secured']).astype(int)
        
        fisd_matched = panel['coupon'].notna().sum()
        fisd_total = len(panel)
        logger.info(f"   FISD match rate: {fisd_matched}/{fisd_total} rows "
                     f"({100*fisd_matched/fisd_total:.1f}%) have bond characteristics.")
    else:
        logger.info("3a. Skipping FISD merge (file not available).")

    # Merge FRED macro shocks and compute target spreads
    
    logger.info("3b. Merging FRED Macro Shocks and Computing Target Spreads...")
    # Merge FRED macro data based on date
    panel = pd.merge(panel, df_fred[['DGS10', 'T10Y2Y', 'VIX']], on='date', how='inner')
    
    # Calculate Individual Bond Credit Spread (Yield - 10Y Treasury)
    panel['CS_i'] = panel['vwap_yield'] - panel['DGS10']
    
    # Sort securely by bond and date to compute daily differenced shocks
    panel = panel.sort_values(['cusip', 'date'])
    
    # Compute stationarized changes for PCMCI / iTransformer
    panel['d_CS_i'] = panel.groupby('cusip')['CS_i'].diff()
    panel['d_DGS10'] = panel.groupby('cusip')['DGS10'].diff()
    panel['d_T10Y2Y'] = panel.groupby('cusip')['T10Y2Y'].diff()
    panel['d_VIX'] = panel.groupby('cusip')['VIX'].diff()
    
    # Drop first days containing NaNs from differencing
    panel = panel.dropna(subset=['d_CS_i', 'leverage'])

    logger.info(f"Master Panel Built: {len(panel)} rows across {panel['cusip'].nunique()} bonds.")
    
    # Column ordering and output

    # Core columns (always present)
    cols = [
        'cusip', 'issuer_cusip', 'date', 'CS_i', 'd_CS_i', 
        'daily_return', 'leverage', 'daily_volume', 'market_cap', 
        'd_DGS10', 'd_T10Y2Y', 'd_VIX',
    ]
    
    # Bond-level features (present only if bond characteristics contain valid data)
    if df_fisd is not None:
        bond_cols = ['coupon', 'ttm_years', 'is_senior']
        active_bond_cols = [c for c in bond_cols if c in panel.columns and panel[c].notna().any()]
        cols.extend(active_bond_cols)
    
    df_final = panel[cols]
    
    output_path = PROCESSED_DIR / "MASTER_MODEL_PANEL.csv"
    df_final.to_csv(output_path, index=False)
    logger.info(f"Saved strictly aligned panel to '{output_path}'")

if __name__ == "__main__":
    build_master_panel()