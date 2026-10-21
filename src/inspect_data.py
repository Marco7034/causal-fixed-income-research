import matplotlib.pyplot as plt
import pandas as pd
from data_pipeline import PipelineConfig, MacroFixedIncomePipeline

# Make pandas print all columns cleanly in the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

if __name__ == "__main__":
    config = PipelineConfig(start_date="2000-01-01", end_date="2026-10-01", frequency="B")
    pipeline = MacroFixedIncomePipeline(config)

    # Fetch, align, and engineer
    df_fred = pipeline.fetch_fred_data()
    df_yf = pipeline.fetch_yf_data()
    df_aligned = pipeline.align_and_clean(df_fred, df_yf)
    df_levels, df_stationary = pipeline.engineer_structural_features(df_aligned)

    # Save to CSV
    df_levels.to_csv("data_levels_2000_2026.csv")
    df_stationary.to_csv("data_stationary_2000_2026.csv")
    print("\n[Saved] 'data_levels_2000_2026.csv' and 'data_stationary_2000_2026.csv'")

    # Print actual data samples and summary statistics in terminal
    print("\n================ RAW LEVELS (First 5 Trading Days) ================")
    print(df_levels.head(5).round(4))

    print("\n================ RAW LEVELS (Last 5 Trading Days) =================")
    print(df_levels.tail(5).round(4))

    print("\n================ RAW LEVELS SUMMARY STATISTICS ====================")
    print(df_levels.describe().round(4))

    print("\n=========== STATIONARIZED SHOCKS (Last 5 Trading Days) ============")
    print(df_stationary.tail(5).round(4))

    print("\n=========== CONTEMPORANEOUS CORRELATION MATRIX (SHOCKS) ===========")
    print(df_stationary[["d_CS_BBB", "d_DGS10", "d_T10Y2Y", "ret_SPY", "d_VIX"]].corr().round(4))

    # Plot the 26-year historical overview across regimes
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(df_levels.index, df_levels["CS_BBB"], color="#1f77b4", lw=1.2)
    axes[0].set_ylabel("BBB Spread (%)")
    axes[0].set_title("26-Year Fixed Income & Macro Regimes (2000–2026): Target & Structural Drivers")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_levels.index, df_levels["DGS10"], label="10Y Treasury (DGS10)", color="#ff7f0e", lw=1.1)
    axes[1].plot(df_levels.index, df_levels["DFF"], label="Fed Funds Rate (DFF)", color="#2ca02c", lw=1.1, alpha=0.8)
    axes[1].set_ylabel("Rate (%)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_levels.index, df_levels["T10Y2Y"], color="#9467bd", lw=1.1)
    axes[2].axhline(0, color="black", linestyle="--", lw=0.8)
    axes[2].set_ylabel("10Y-2Y Slope (%)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df_levels.index, df_levels["VIX"], color="#d62728", lw=1.0)
    axes[3].set_ylabel("VIX Index")
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("credit_spread_overview.png", dpi=200)
    print("\n[Saved Plot] 'credit_spread_overview.png'")