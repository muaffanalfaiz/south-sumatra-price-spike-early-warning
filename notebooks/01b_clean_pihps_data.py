"""
01d_clean_pihps_data.py — Clean and reshape raw PIHPS price data
================================================================
Transforms the wide-format API output (dates as columns) into the
long-format panel dataset the modeling pipeline expects.

INPUT:  data/raw/pihps_prices_all.csv  (from 01c scraper)
OUTPUT: data/processed/real_prices.csv  (ready for 05_modeling_pipeline.py)

USAGE:
    python scripts/01d_clean_pihps_data.py
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
PROC_DIR = PROJECT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Map PIHPS categories to specific commodity names
CATEGORY_TO_COMMODITY = {
    "Bawang Merah": "Bawang Merah Ukuran Sedang",
    "Bawang Putih": "Bawang Putih Ukuran Sedang",
    "Cabai Merah": "Cabai Merah Besar + Keriting",  # category combines both
    "Cabai Rawit": "Cabai Rawit Merah + Hijau",      # category combines both
}


def main():
    log.info("=" * 60)
    log.info("Cleaning PIHPS Price Data")
    log.info("=" * 60)

    # ── Load raw data ──
    raw_path = RAW_DIR / "pihps_prices_all.csv"
    df = pd.read_csv(raw_path, encoding="utf-8-sig", low_memory=False)
    log.info(f"Raw data loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # ── Identify date columns vs metadata columns ──
    meta_cols = ["no", "name", "level", "_comcat_id", "_category", "_start_date", "_end_date", "_province"]
    date_cols = [c for c in df.columns if c not in meta_cols]
    log.info(f"Date columns: {len(date_cols)}")

    # ── Filter to useful rows ──
    # level=2 is kabupaten/kota, level=1 is province aggregate
    # Keep both — province aggregate is useful as a reference
    df_filtered = df[df["level"].isin([1, 2])].copy()
    log.info(f"After filtering (level 1+2): {len(df_filtered)} rows")
    log.info(f"Locations: {df_filtered['name'].unique().tolist()}")
    log.info(f"Categories: {df_filtered['_category'].unique().tolist()}")

    # ── Melt wide to long format ──
    log.info("Melting wide format to long...")
    df_long = df_filtered.melt(
        id_vars=["name", "level", "_comcat_id", "_category"],
        value_vars=date_cols,
        var_name="date_str",
        value_name="price_raw",
    )

    # Drop rows where price is empty/NaN
    df_long = df_long.dropna(subset=["price_raw"])
    df_long = df_long[df_long["price_raw"] != ""]
    df_long = df_long[df_long["price_raw"] != "-"]
    log.info(f"After melt + drop empty: {len(df_long)} rows")

    # ── Parse dates ──
    # Format is DD/MM/YYYY
    df_long["date"] = pd.to_datetime(df_long["date_str"], format="%d/%m/%Y", errors="coerce")
    bad_dates = df_long["date"].isna().sum()
    if bad_dates > 0:
        log.warning(f"  {bad_dates} unparseable dates dropped")
        df_long = df_long.dropna(subset=["date"])

    # ── Parse prices ──
    # Indonesian format: "43,750" means 43750 (comma = thousands separator)
    df_long["price_rp_kg"] = (
        df_long["price_raw"]
        .astype(str)
        .str.replace(",", "", regex=False)  # remove thousands separator
        .str.replace(".", "", regex=False)  # in case of period separators
        .str.strip()
    )
    # Convert to numeric
    df_long["price_rp_kg"] = pd.to_numeric(df_long["price_rp_kg"], errors="coerce")
    bad_prices = df_long["price_rp_kg"].isna().sum()
    if bad_prices > 0:
        log.warning(f"  {bad_prices} unparseable prices dropped")
        df_long = df_long.dropna(subset=["price_rp_kg"])

    # ── Rename and clean columns ──
    df_long = df_long.rename(columns={
        "name": "district",
        "_category": "commodity",
    })
    df_long["commodity"] = df_long["commodity"].map(CATEGORY_TO_COMMODITY).fillna(df_long["commodity"])

    # Keep only what we need
    df_clean = df_long[["date", "district", "commodity", "price_rp_kg", "level"]].copy()
    df_clean = df_clean.sort_values(["district", "commodity", "date"]).reset_index(drop=True)

    log.info(f"\nCleaned dataset: {len(df_clean)} rows")
    log.info(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    log.info(f"Districts: {df_clean['district'].unique().tolist()}")
    log.info(f"Commodities: {df_clean['commodity'].unique().tolist()}")

    # ── Resample to weekly (Monday-aligned) ──
    log.info("\nResampling daily to weekly (median)...")
    df_clean["week"] = df_clean["date"].dt.to_period("W-MON").dt.start_time

    weekly = (df_clean
        .groupby(["week", "district", "commodity", "level"])["price_rp_kg"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "week": "date",
            "median": "price_rp_kg",
            "mean": "price_mean",
            "std": "price_std",
            "count": "obs_count",
        })
    )
    log.info(f"Weekly dataset: {len(weekly)} rows")
    log.info(f"Weeks: {weekly['date'].nunique()}")

    # ── Compute spike detection (z-score) ──
    log.info("Computing price spike indicators...")
    weekly = weekly.sort_values(["district", "commodity", "date"])

    weekly["rolling_med"] = (weekly
        .groupby(["district", "commodity"])["price_rp_kg"]
        .transform(lambda x: x.rolling(8, min_periods=4).median()))
    weekly["rolling_std"] = (weekly
        .groupby(["district", "commodity"])["price_rp_kg"]
        .transform(lambda x: x.rolling(8, min_periods=4).std()))
    weekly["z_score"] = ((weekly["price_rp_kg"] - weekly["rolling_med"]) / weekly["rolling_std"]).fillna(0)
    weekly["spike"] = (weekly["z_score"] > 1.5).astype(int)

    spike_count = weekly["spike"].sum()
    spike_rate = weekly["spike"].mean() * 100
    log.info(f"Spikes detected: {spike_count} ({spike_rate:.1f}%)")

    # ── Save ──
    # Daily clean data
    daily_path = PROC_DIR / "real_prices_daily.csv"
    df_clean["date"] = df_clean["date"].dt.strftime("%Y-%m-%d")
    df_clean.to_csv(daily_path, index=False)
    log.info(f"\nDaily data saved: {daily_path}")

    # Weekly data (this is what the pipeline uses)
    weekly_path = PROC_DIR / "real_prices.csv"
    weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
    weekly.to_csv(weekly_path, index=False)
    log.info(f"Weekly data saved: {weekly_path}")

    # ── Summary stats ──
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    for district in weekly["district"].unique():
        for commodity in weekly["commodity"].unique():
            subset = weekly[(weekly["district"] == district) & (weekly["commodity"] == commodity)]
            if len(subset) == 0:
                continue
            spikes = subset["spike"].sum()
            med_price = subset["price_rp_kg"].median()
            log.info(f"  {district:25s} | {commodity:30s} | weeks: {len(subset):3d} | median: Rp {med_price:,.0f} | spikes: {spikes}")

    log.info(f"\nIMPORTANT NOTE:")
    log.info(f"PIHPS only monitors 2 cities in South Sumatra: Palembang and Lubuk Linggau.")
    log.info(f"The province-level aggregate ('Sumatera Selatan') is included as a 3rd location.")
    log.info(f"This is an honest data constraint — document it in the report.")
    log.info(f"\nNext step: Run the GEE script, then use 05_modeling_pipeline.py with real data.")


if __name__ == "__main__":
    main()
