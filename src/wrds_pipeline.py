import logging
import pandas as pd
import wrds

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

class WRDSMicroPipeline:
    def __init__(self):
        """Initializes connection using the cached ~/.pgpass credentials."""
        logger.info("Connecting to WRDS")
        self.db = wrds.Connection()

    def fetch_trace_yields(self, start_date: str, end_date: str, limit: int = 250) -> pd.DataFrame:
        """
        Pulls daily secondary market pricing and yields from TRACE.
        Filters for the most actively traded bonds to ensure a dense, continuous time-series.
        """
        logger.info(f"Fetching daily TRACE yields from {start_date} to {end_date}...")
        
        # Server-side volume-weighted aggregation for the top 'limit' most liquid bonds
        # Note: WRDS uses 'cusip_id' in trace_enhanced, not 'cusip'
        query = f"""
            WITH liquid_bonds AS (
                SELECT cusip_id, COUNT(DISTINCT trd_exctn_dt) as active_days
                FROM trace.trace_enhanced
                WHERE trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'
                  AND trc_st = 'T' 
                  AND yld_pt IS NOT NULL
                GROUP BY cusip_id
                ORDER BY active_days DESC
                LIMIT {limit}
            )
            SELECT t.cusip_id AS cusip, 
                   t.trd_exctn_dt AS date, 
                   SUM(t.entrd_vol_qt) AS daily_volume,
                   SUM(t.yld_pt * t.entrd_vol_qt) / SUM(t.entrd_vol_qt) AS vwap_yield
            FROM trace.trace_enhanced t
            JOIN liquid_bonds l ON t.cusip_id = l.cusip_id
            WHERE t.trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'
              AND t.trc_st = 'T'
              AND t.yld_pt IS NOT NULL
            GROUP BY t.cusip_id, t.trd_exctn_dt
            ORDER BY t.cusip_id, t.trd_exctn_dt
        """
        df_trace = self.db.raw_sql(query)
        df_trace['date'] = pd.to_datetime(df_trace['date'])
        
        # Extract the 6-digit issuer CUSIP to link to CRSP/Compustat
        df_trace['issuer_cusip'] = df_trace['cusip'].str[:6]
        
        logger.info(f"Retrieved {len(df_trace)} daily pricing rows for {df_trace['cusip'].nunique()} unique bonds.")
        return df_trace

    def fetch_crsp_equity(self, issuer_cusips: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Pulls daily firm-level equity returns from CRSP matching our TRACE bond issuers.
        Provides the firm-specific asset value shock for the Merton structural model.
        """
        logger.info("Fetching CRSP Daily Equity Returns...")
        
        # Format the 6-digit CUSIPs for the SQL IN clause
        cusip_str = "','".join(issuer_cusips)
        
        # CRSP uses an 8-digit cusip in the dsf table, we match on the first 6 digits (the issuer)
        query = f"""
            SELECT SUBSTR(cusip, 1, 6) AS issuer_cusip, 
                   date, 
                   ret AS daily_return, 
                   vol AS equity_volume,
                   ABS(prc) * shrout AS market_cap
            FROM crsp.dsf
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
              AND SUBSTR(cusip, 1, 6) IN ('{cusip_str}')
              AND ret IS NOT NULL
        """
        df_crsp = self.db.raw_sql(query)
        df_crsp['date'] = pd.to_datetime(df_crsp['date'])
        
        # Drop duplicates in case a firm has multiple equity share classes, keep the largest by market cap
        df_crsp = df_crsp.sort_values('market_cap', ascending=False).drop_duplicates(subset=['issuer_cusip', 'date'])
        
        logger.info(f"Retrieved {len(df_crsp)} daily equity rows.")
        return df_crsp

    def fetch_compustat_leverage(self, issuer_cusips: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Pulls quarterly firm fundamentals from Compustat - Capital IQ.
        Calculates firm leverage (Total Debt / Total Assets).
        
        Includes `rdq` (Report Date of Quarterly earnings) to enable the
        filing-date lag correction in build_master_panel.py, preventing
        lookahead bias from using datadate (fiscal period end) as the
        availability date.
        """
        logger.info("Fetching Compustat Quarterly Fundamentals (with filing dates)...")
        
        cusip_str = "','".join(issuer_cusips)
        
        query = f"""
            SELECT SUBSTR(cusip, 1, 6) AS issuer_cusip, 
                   datadate AS fiscal_date,
                   rdq AS filing_date,
                   tic AS ticker,
                   dlttq + dlcq AS total_debt, 
                   atq AS total_assets
            FROM comp.fundq
            WHERE datadate BETWEEN '{start_date}' AND '{end_date}'
              AND SUBSTR(cusip, 1, 6) IN ('{cusip_str}')
              AND indfmt = 'INDL' 
              AND datafmt = 'STD' 
              AND popsrc = 'D' 
              AND consol = 'C'
        """
        df_comp = self.db.raw_sql(query)
        df_comp['fiscal_date'] = pd.to_datetime(df_comp['fiscal_date'])
        df_comp['filing_date'] = pd.to_datetime(df_comp['filing_date'])
        
        # Use the actual SEC filing date (rdq) as the availability date.
        # If rdq is missing for some rows, fall back to fiscal_date + 45 calendar days
        # (conservative estimate of large-accelerated-filer 10-Q deadline).
        df_comp['date'] = df_comp['filing_date'].fillna(
            df_comp['fiscal_date'] + pd.offsets.Day(45)
        )
        
        # Calculate Leverage Ratio (avoiding division by zero)
        df_comp['leverage'] = df_comp['total_debt'] / df_comp['total_assets'].replace(0, pd.NA)
        
        # Drop intra-quarter duplicates (keep last filing if restated)
        df_comp = df_comp.sort_values('date').drop_duplicates(
            subset=['issuer_cusip', 'fiscal_date'], keep='last'
        )
        
        logger.info(f"Retrieved {len(df_comp)} quarterly fundamental rows "
                     f"({df_comp['filing_date'].notna().sum()} with actual rdq filing dates).")
        return df_comp

    def fetch_bond_characteristics(self, cusips: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Pulls bond-level characteristics for each 9-digit CUSIP.
        
        Strategy:
        1. Try FISD (fisd.fisd_mergedissue) for coupon, maturity, seniority — the gold standard.
        2. If FISD access is denied, fall back to TRACE Enhanced metadata
           (bond_sym_id, scrty_type_cd) which is available without extra subscription.
           Coupon and maturity will be unavailable in this case.
        """
        cusip_str = "','".join(cusips)
        
        # FISD
        try:
            logger.info("Attempting FISD bond characteristics (coupon, maturity, seniority)")
            query = f"""
                SELECT complete_cusip AS cusip,
                       coupon,
                       maturity AS maturity_date,
                       offering_date,
                       security_level,
                       bond_type
                FROM fisd.fisd_mergedissue
                WHERE complete_cusip IN ('{cusip_str}')
            """
            df = self.db.raw_sql(query)
            df['maturity_date'] = pd.to_datetime(df['maturity_date'])
            df['offering_date'] = pd.to_datetime(df['offering_date'])
            
            seniority_map = {
                'SEN': 'Senior', 'SS': 'Senior Secured', 'SB': 'Senior Subordinated',
                'SUB': 'Subordinated', 'JUN': 'Junior', 'SENB': 'Senior Bank',
            }
            df['seniority'] = df['security_level'].map(seniority_map).fillna('Other')
            
            logger.info(f"FISD: Retrieved characteristics for {len(df)} bonds "
                         f"({df['coupon'].notna().sum()} with coupon data).")
            return df
            
        except Exception as e:
            logger.warning(f"FISD access denied ({type(e).__name__}). "
                           "Falling back to TRACE Enhanced metadata.")
        
        # TRACE Enhanced metadata
        logger.info("Fetching bond metadata from TRACE Enhanced (bond_sym_id, scrty_type_cd)")
        query = f"""
            SELECT cusip_id AS cusip,
                   MODE() WITHIN GROUP (ORDER BY bond_sym_id) AS bond_symbol,
                   MODE() WITHIN GROUP (ORDER BY scrty_type_cd) AS security_type,
                   MODE() WITHIN GROUP (ORDER BY sub_prdct) AS sub_product,
                   MIN(trd_exctn_dt) AS first_trade_date,
                   MAX(trd_exctn_dt) AS last_trade_date
            FROM trace.trace_enhanced
            WHERE cusip_id IN ('{cusip_str}')
              AND trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'
              AND trc_st = 'T'
            GROUP BY cusip_id
        """
        df = self.db.raw_sql(query)
        
        # Without FISD, coupon and maturity are unavailable — set to NaN
        df['coupon'] = pd.NA
        df['maturity_date'] = pd.NaT
        df['seniority'] = 'Senior'  # Conservative default for liquid TRACE bonds
        
        logger.info(f"TRACE metadata: Retrieved info for {len(df)} bonds. ")
        return df

if __name__ == "__main__":
    from pathlib import Path
    
    RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    pipeline = WRDSMicroPipeline()
    
    # Define the timeframe (Test a robust 5-year window tracking the COVID shock and Tightening regime)
    START = "2019-01-01" 
    END = "2024-01-01"
    
    # Extract Data sequentially so CRSP/Compustat only pull data for the bonds we actually found in TRACE
    df_trace = pipeline.fetch_trace_yields(START, END, limit=250)
    
    unique_issuers = df_trace['issuer_cusip'].dropna().unique().tolist()
    unique_bonds = df_trace['cusip'].dropna().unique().tolist()
    
    df_crsp = pipeline.fetch_crsp_equity(unique_issuers, START, END)
    df_compustat = pipeline.fetch_compustat_leverage(unique_issuers, START, END)
    df_bond_chars = pipeline.fetch_bond_characteristics(unique_bonds, START, END)
    
    # Save raw micro data locally
    df_trace.to_csv(RAW_DIR / "wrds_trace_yields.csv", index=False)
    df_crsp.to_csv(RAW_DIR / "wrds_crsp_equity.csv", index=False)
    df_compustat.to_csv(RAW_DIR / "wrds_compustat_leverage.csv", index=False)
    df_bond_chars.to_csv(RAW_DIR / "wrds_bond_chars.csv", index=False)
    
    print(f"\nWRDS extraction complete. Micro-data saved to {RAW_DIR}/")