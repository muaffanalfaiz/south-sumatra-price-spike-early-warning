"""
04_generate_synthetic.py — Generate realistic synthetic data for testing
=========================================================================
Creates fake-but-structurally-realistic datasets that match what the real
data from PIHPS, GEE, and OSM will look like. This lets you test the
entire pipeline end-to-end before you have real data.

USAGE:
    python 04_generate_synthetic.py

OUTPUT:
    data/processed/synthetic_prices.csv
    data/processed/synthetic_ndvi.csv
    data/processed/synthetic_rainfall.csv
    data/processed/synthetic_road_distances.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

np.random.seed(42)

# ── Config ──
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Districts in South Sumatra
DISTRICTS = [
    "Palembang", "Ogan Komering Ilir", "Ogan Komering Ulu", "Muara Enim",
    "Lahat", "Musi Rawas", "Musi Banyuasin", "Banyuasin", "OKU Selatan",
    "OKU Timur", "Ogan Ilir", "Empat Lawang", "PALI",
    "Musi Rawas Utara", "Lubuklinggau", "Prabumulih", "Pagar Alam"
]

PRODUCING = ["OKU Selatan", "Lahat", "Pagar Alam", "Empat Lawang", "Ogan Komering Ulu", "OKU Timur"]
CONSUMING = [d for d in DISTRICTS if d not in PRODUCING]

COMMODITIES = ["Cabai Rawit Merah", "Cabai Merah Keriting", "Bawang Merah", "Tomat"]

# Base prices (Rp/kg) and volatility by commodity
COMMODITY_PARAMS = {
    "Cabai Rawit Merah":  {"base_price": 55000, "volatility": 0.35, "spike_freq": 0.08},
    "Cabai Merah Keriting": {"base_price": 45000, "volatility": 0.30, "spike_freq": 0.06},
    "Bawang Merah":       {"base_price": 38000, "volatility": 0.20, "spike_freq": 0.04},
    "Tomat":              {"base_price": 12000, "volatility": 0.25, "spike_freq": 0.05},
}

DATE_RANGE = pd.date_range("2023-01-01", "2025-12-28", freq="W-MON")


def generate_prices():
    """
    Generate weekly price data with realistic seasonal patterns and spikes.
    
    Key realism features:
    - Seasonal cycle (prices peak in wet season Dec-Feb)
    - Random spikes that are more likely when NDVI is low
    - Spatial correlation (nearby districts have similar prices)
    - Consuming districts have higher prices than producing ones
    """
    log.info("Generating synthetic price data...")
    rows = []

    for district in DISTRICTS:
        is_producer = district in PRODUCING
        # Producing districts have lower base prices (less transport markup)
        price_markup = 1.0 if is_producer else np.random.uniform(1.10, 1.30)
        # District-level random effect
        district_effect = np.random.normal(1.0, 0.05)

        for commodity in COMMODITIES:
            params = COMMODITY_PARAMS[commodity]
            base = params["base_price"] * price_markup * district_effect
            vol = params["volatility"]
            spike_freq = params["spike_freq"]

            for week in DATE_RANGE:
                # Seasonal component: peak in Dec-Feb (wet season disrupts supply)
                month = week.month
                if month in [12, 1, 2]:
                    seasonal = 1.20 + np.random.normal(0, 0.05)
                elif month in [3, 4, 5]:
                    seasonal = 1.05 + np.random.normal(0, 0.03)
                elif month in [6, 7, 8]:
                    seasonal = 0.85 + np.random.normal(0, 0.03)
                else:  # Sep-Nov
                    seasonal = 0.95 + np.random.normal(0, 0.04)

                # Random noise
                noise = np.random.normal(0, vol * 0.3)

                # Spike injection (more likely for consuming districts in wet season)
                spike = 0
                spike_prob = spike_freq
                if month in [12, 1, 2]:
                    spike_prob *= 2.5  # wet season spikes
                if not is_producer:
                    spike_prob *= 1.5  # consuming districts spike more

                if np.random.random() < spike_prob:
                    spike = np.random.uniform(0.3, 0.8)  # 30-80% price jump

                price = base * seasonal * (1 + noise + spike)
                price = max(price, base * 0.4)  # floor

                rows.append({
                    "date": week.strftime("%Y-%m-%d"),
                    "district": district,
                    "commodity": commodity,
                    "price_rp_kg": round(price),
                    "is_spike_injected": 1 if spike > 0.2 else 0,  # ground truth for validation
                })

    df = pd.DataFrame(rows)
    
    # Compute the z-score based spike detection (what the real pipeline does)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["district", "commodity", "date"])
    
    df["rolling_med"] = (df.groupby(["district", "commodity"])["price_rp_kg"]
        .transform(lambda x: x.rolling(8, min_periods=4).median()))
    df["rolling_std"] = (df.groupby(["district", "commodity"])["price_rp_kg"]
        .transform(lambda x: x.rolling(8, min_periods=4).std()))
    df["z_score"] = ((df["price_rp_kg"] - df["rolling_med"]) / df["rolling_std"]).fillna(0)
    df["spike"] = (df["z_score"] > 1.5).astype(int)
    
    # Convert date back to string for CSV
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    path = PROC_DIR / "synthetic_prices.csv"
    df.to_csv(path, index=False)
    
    spike_rate = df["spike"].mean() * 100
    log.info(f"  Prices: {len(df)} rows, spike rate: {spike_rate:.1f}%")
    log.info(f"  Saved to {path}")
    return df


def generate_ndvi():
    """
    Generate synthetic NDVI anomaly data matching GEE export structure.
    
    Key realism features:
    - Seasonal NDVI cycle (lower in dry season Jun-Sep)
    - Negative anomalies correlate with subsequent price spikes
    - 16-day compositing windows matching MODIS MOD13Q1
    - Values in MODIS scale (x10000)
    """
    log.info("Generating synthetic NDVI anomaly data...")
    
    # 16-day composite dates (23 per year)
    dates_16day = pd.date_range("2023-01-01", "2025-12-31", freq="16D")
    
    rows = []
    for district in DISTRICTS:
        is_producer = district in PRODUCING
        # Producers have higher baseline NDVI (agricultural land)
        base_ndvi = 6000 if is_producer else 5000  # MODIS scale

        for date in dates_16day:
            month = date.month
            # Seasonal NDVI: higher in wet season, lower in dry season
            if month in [12, 1, 2, 3]:
                seasonal_ndvi = base_ndvi * 1.15
            elif month in [4, 5]:
                seasonal_ndvi = base_ndvi * 1.05
            elif month in [6, 7, 8, 9]:
                seasonal_ndvi = base_ndvi * 0.80
            else:
                seasonal_ndvi = base_ndvi * 0.95

            # Random variation
            ndvi = seasonal_ndvi + np.random.normal(0, 400)

            # Anomaly (current - baseline)
            # Inject negative anomalies that correlate with price spikes
            anomaly = np.random.normal(0, 300)

            # For producing districts in certain months, inject stress events
            if is_producer and np.random.random() < 0.07:
                anomaly = np.random.uniform(-1500, -600)  # crop stress event

            doy = date.timetuple().tm_yday
            doy_window = doy // 16

            rows.append({
                "ADM2_NAME": district,
                "date": date.strftime("%Y-%m-%d"),
                "doy_window": doy_window,
                "NDVI": round(ndvi),
                "EVI": round(ndvi * 0.6),  # EVI is typically ~0.6x NDVI
                "NDVI_anomaly": round(anomaly),
                "EVI_anomaly": round(anomaly * 0.6),
            })

    df = pd.DataFrame(rows)
    path = PROC_DIR / "synthetic_ndvi.csv"
    df.to_csv(path, index=False)
    log.info(f"  NDVI: {len(df)} rows, saved to {path}")
    return df


def generate_rainfall():
    """
    Generate synthetic weekly rainfall data matching CHIRPS export structure.
    """
    log.info("Generating synthetic rainfall data...")
    
    rows = []
    for district in DISTRICTS:
        for week in DATE_RANGE:
            month = week.month
            # Wet season (Nov-Mar): heavy rainfall
            if month in [11, 12, 1, 2, 3]:
                mean_rain = np.random.uniform(40, 120)  # mm/week
            elif month in [4, 5, 10]:
                mean_rain = np.random.uniform(20, 60)
            else:  # Jun-Sep dry
                mean_rain = np.random.uniform(5, 25)
            
            # Occasionally inject extreme rainfall events
            if np.random.random() < 0.05 and month in [11, 12, 1, 2, 3]:
                mean_rain *= np.random.uniform(2.0, 3.5)
            
            rain = max(0, mean_rain + np.random.normal(0, 10))
            
            rows.append({
                "ADM2_NAME": district,
                "week_start": week.strftime("%Y-%m-%d"),
                "mean": round(rain, 1),
            })
    
    df = pd.DataFrame(rows)
    path = PROC_DIR / "synthetic_rainfall.csv"
    df.to_csv(path, index=False)
    log.info(f"  Rainfall: {len(df)} rows, saved to {path}")
    return df


def generate_road_distances():
    """
    Generate synthetic inter-district distance features.
    """
    log.info("Generating synthetic road distance features...")
    
    rows = []
    for district in DISTRICTS:
        if district in PRODUCING:
            min_dist = 0
            min_time = 0
            nearest = district
        else:
            min_dist = np.random.uniform(50, 250)
            min_time = min_dist / 35  # ~35 km/h average
            nearest = np.random.choice(PRODUCING)
        
        rows.append({
            "district": district,
            "nearest_producer": nearest,
            "min_distance_km": round(min_dist, 1),
            "min_travel_hrs": round(min_time, 2),
            "avg_distance_km": round(min_dist * 1.3, 1),
        })
    
    df = pd.DataFrame(rows)
    path = PROC_DIR / "synthetic_road_distances.csv"
    df.to_csv(path, index=False)
    log.info(f"  Distances: {len(df)} rows, saved to {path}")
    return df


def main():
    log.info("=" * 60)
    log.info("Generating Synthetic Datasets for Pipeline Testing")
    log.info("=" * 60)
    
    prices = generate_prices()
    ndvi = generate_ndvi()
    rainfall = generate_rainfall()
    distances = generate_road_distances()
    
    log.info("\n=== Summary ===")
    log.info(f"Prices:    {len(prices):>6,} rows | {prices['spike'].sum()} spikes ({prices['spike'].mean()*100:.1f}%)")
    log.info(f"NDVI:      {len(ndvi):>6,} rows | {ndvi['NDVI_anomaly'].lt(-500).sum()} stress events")
    log.info(f"Rainfall:  {len(rainfall):>6,} rows | {rainfall['mean'].gt(150).sum()} extreme weeks")
    log.info(f"Distances: {len(distances):>6,} rows")
    log.info(f"\nAll files saved to {PROC_DIR}")
    log.info("Next step: python 05_modeling_pipeline.py --synthetic")


if __name__ == "__main__":
    main()
