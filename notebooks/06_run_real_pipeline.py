"""
06_run_real_pipeline.py — Full pipeline on real PIHPS + GEE data
================================================================
Merges the cleaned PIHPS prices with GEE NDVI anomalies and rainfall,
computes road distances, engineers features, and trains models.

REQUIRES (run these first):
    1. python scripts/01c_scrape_pihps_final.py  (scrape prices)
    2. python scripts/01d_clean_pihps_data.py     (clean prices)
    3. GEE exports in data/raw/:
       - sumsel_ndvi_anomaly.csv
       - sumsel_weekly_rainfall.csv

USAGE:
    python scripts/06_run_real_pipeline.py

OUTPUT:
    data/processed/real_panel_dataset.csv
    outputs/real_model_evaluation.txt
    outputs/real_risk_map.html
    outputs/real_*.png (charts)
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
PROC_DIR = PROJECT / "data" / "processed"
OUTPUT_DIR = PROJECT / "outputs"
PROC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Highland producing districts (supply-side NDVI signal)
PRODUCING_DISTRICTS_GEE = [
    "Ogan Komering Ulu Selatan", "Lahat", "Kota Pagaralam",
    "Ogan Komering Ulu", "Ogan Komering Ulu Timur"
]

# Name mapping: PIHPS -> GEE
PIHPS_TO_GEE = {
    "Kota Palembang": "Kota Palembang",
    "Kota Lubuk Linggau": "Kota Lubuklinggau",
}

# Approximate distances from producing highlands to consuming markets
SUPPLY_DISTANCES = {
    "Kota Palembang": {"nearest_producer": "OKU Timur", "min_distance_km": 280, "min_travel_hrs": 6.5},
    "Kota Lubuk Linggau": {"nearest_producer": "Lahat", "min_distance_km": 75, "min_travel_hrs": 2.0},
    "Sumatera Selatan": {"nearest_producer": "province_avg", "min_distance_km": 150, "min_travel_hrs": 4.0},
}


# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════

def load_all():
    log.info("Loading all data sources...")

    # Prices (cleaned weekly)
    prices = pd.read_csv(PROC_DIR / "real_prices.csv", parse_dates=["date"])
    log.info(f"  Prices: {len(prices)} rows | {prices['district'].nunique()} locations | {prices['commodity'].nunique()} commodities")

    # NDVI
    ndvi = pd.read_csv(RAW_DIR / "sumsel_ndvi_anomaly.csv", parse_dates=["date"])
    for col in ["NDVI", "EVI", "NDVI_anomaly", "EVI_anomaly"]:
        ndvi[col] = ndvi[col] / 10000.0  # MODIS scale to actual
    log.info(f"  NDVI: {len(ndvi)} rows | {ndvi['ADM2_NAME'].nunique()} districts")

    # Rainfall
    rain = pd.read_csv(RAW_DIR / "sumsel_weekly_rainfall.csv", parse_dates=["week_start"])
    rain = rain.rename(columns={"mean": "rainfall_mm", "week_start": "date"})
    log.info(f"  Rainfall: {len(rain)} rows | {rain['ADM2_NAME'].nunique()} districts")

    return prices, ndvi, rain


# ══════════════════════════════════════════════════════════
# 2. BUILD SUPPLY-SIDE NDVI (from producing districts)
# ══════════════════════════════════════════════════════════

def build_supply_ndvi(ndvi):
    """
    Compute the average NDVI anomaly across producing districts.
    This is the "harvest stress" signal — when producers show
    negative NDVI anomaly, prices in consuming markets spike later.
    """
    log.info("Building supply-side NDVI signal from producing districts...")

    producers = ndvi[ndvi["ADM2_NAME"].isin(PRODUCING_DISTRICTS_GEE)].copy()
    log.info(f"  Producing districts found: {sorted(producers['ADM2_NAME'].unique())}")

    # Average across all producing districts per date
    supply_ndvi = (producers
        .groupby("date")
        .agg(
            supply_ndvi_anomaly=("NDVI_anomaly", "mean"),
            supply_evi_anomaly=("EVI_anomaly", "mean"),
            supply_ndvi=("NDVI", "mean"),
        )
        .reset_index()
    )

    # Resample to weekly (Monday-aligned) via forward-fill
    supply_ndvi = supply_ndvi.set_index("date").resample("W-MON").ffill().reset_index()
    log.info(f"  Supply NDVI weekly: {len(supply_ndvi)} rows")

    return supply_ndvi


def build_local_ndvi(ndvi):
    """Build NDVI for each PIHPS market location."""
    log.info("Building local NDVI for PIHPS market locations...")

    frames = []
    for pihps_name, gee_name in PIHPS_TO_GEE.items():
        local = ndvi[ndvi["ADM2_NAME"] == gee_name].copy()
        local["district"] = pihps_name
        local = local.set_index("date").resample("W-MON").ffill().reset_index()
        local = local.rename(columns={
            "NDVI_anomaly": "local_ndvi_anomaly",
            "EVI_anomaly": "local_evi_anomaly",
            "NDVI": "local_ndvi",
        })
        frames.append(local[["date", "district", "local_ndvi_anomaly", "local_evi_anomaly", "local_ndvi"]])

    # Province average for "Sumatera Selatan"
    prov_avg = (ndvi
        .groupby("date")
        .agg(local_ndvi_anomaly=("NDVI_anomaly", "mean"),
             local_evi_anomaly=("EVI_anomaly", "mean"),
             local_ndvi=("NDVI", "mean"))
        .reset_index())
    prov_avg["district"] = "Sumatera Selatan"
    prov_avg = prov_avg.set_index("date").resample("W-MON").ffill().reset_index()
    frames.append(prov_avg)

    local_ndvi = pd.concat(frames, ignore_index=True)
    log.info(f"  Local NDVI weekly: {len(local_ndvi)} rows")
    return local_ndvi


def build_local_rainfall(rain):
    """Build weekly rainfall for each PIHPS market location."""
    log.info("Building local rainfall for PIHPS market locations...")

    frames = []
    for pihps_name, gee_name in PIHPS_TO_GEE.items():
        local = rain[rain["ADM2_NAME"] == gee_name].copy()
        local["district"] = pihps_name
        frames.append(local[["date", "district", "rainfall_mm"]])

    # Province average
    prov_rain = rain.groupby("date")["rainfall_mm"].mean().reset_index()
    prov_rain["district"] = "Sumatera Selatan"
    frames.append(prov_rain)

    local_rain = pd.concat(frames, ignore_index=True)
    log.info(f"  Local rainfall weekly: {len(local_rain)} rows")
    return local_rain


# ══════════════════════════════════════════════════════════
# 3. MERGE AND ENGINEER FEATURES
# ══════════════════════════════════════════════════════════

def build_panel(prices, supply_ndvi, local_ndvi, local_rain):
    log.info("Building panel dataset...")

    # Align all dates to the same weekly anchor
    # Prices use Tuesday-anchored weeks, NDVI/rain use Monday-anchored
    # Solution: round all dates to the nearest Monday
    prices["date"] = prices["date"].dt.to_period("W").dt.start_time
    supply_ndvi["date"] = supply_ndvi["date"].dt.to_period("W").dt.start_time
    local_ndvi["date"] = local_ndvi["date"].dt.to_period("W").dt.start_time
    local_rain["date"] = local_rain["date"].dt.to_period("W").dt.start_time

    log.info(f"  Date alignment — prices: {prices['date'].iloc[0].date()}, NDVI: {supply_ndvi['date'].iloc[0].date()}")

    # Merge prices + supply NDVI (same for all districts — it's the producer signal)
    panel = prices.merge(supply_ndvi, on="date", how="left")

    # Merge local NDVI (specific to each market)
    panel = panel.merge(local_ndvi, on=["date", "district"], how="left")

    # Merge local rainfall
    panel = panel.merge(local_rain, on=["date", "district"], how="left")

    # Add road distances (static)
    for district, dist_info in SUPPLY_DISTANCES.items():
        mask = panel["district"] == district
        panel.loc[mask, "min_distance_km"] = dist_info["min_distance_km"]
        panel.loc[mask, "min_travel_hrs"] = dist_info["min_travel_hrs"]

    # Sort for lagging
    panel = panel.sort_values(["district", "commodity", "date"])

    # Lagged features (supply-side NDVI from producing areas)
    log.info("  Creating lagged features...")
    for lag in [1, 2, 3, 4]:
        panel[f"supply_ndvi_lag{lag}"] = (
            panel.groupby(["district", "commodity"])["supply_ndvi_anomaly"].shift(lag))
        panel[f"local_ndvi_lag{lag}"] = (
            panel.groupby(["district", "commodity"])["local_ndvi_anomaly"].shift(lag))
        panel[f"rain_lag{lag}"] = (
            panel.groupby(["district", "commodity"])["rainfall_mm"].shift(lag))

    # Rolling features
    panel["rain_4wk_sum"] = (panel.groupby(["district", "commodity"])["rainfall_mm"]
        .transform(lambda x: x.rolling(4, min_periods=2).sum()))
    panel["supply_ndvi_4wk_trend"] = (panel.groupby(["district", "commodity"])["supply_ndvi_anomaly"]
        .transform(lambda x: x.rolling(4, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0)))

    # Calendar
    panel["month"] = panel["date"].dt.month
    panel["is_wet_season"] = panel["month"].isin([11, 12, 1, 2, 3]).astype(int)

    # Flood risk
    panel["flood_risk"] = (panel["rainfall_mm"] > 100).astype(int)
    panel["flood_risk_lag1"] = panel.groupby(["district", "commodity"])["flood_risk"].shift(1)

    # Interaction: supply stress × distance
    panel["stress_x_distance"] = panel["supply_ndvi_lag2"] * panel["min_distance_km"]

    # Price momentum
    panel["price_lag1"] = panel.groupby(["district", "commodity"])["price_rp_kg"].shift(1)
    panel["price_pct_change"] = (panel["price_rp_kg"] - panel["price_lag1"]) / panel["price_lag1"]

    # Drop NaN from lagging
    before = len(panel)
    panel = panel.dropna(subset=["supply_ndvi_lag2", "rain_lag1", "spike"])
    log.info(f"  Panel: {before} -> {len(panel)} rows")

    return panel


# ══════════════════════════════════════════════════════════
# 4. TRAIN AND EVALUATE
# ══════════════════════════════════════════════════════════

FEATURE_COLS = [
    "supply_ndvi_lag2", "supply_ndvi_lag3", "supply_ndvi_lag4",
    "local_ndvi_lag2", "local_ndvi_lag3",
    "rain_lag1", "rain_lag2", "rain_4wk_sum",
    "supply_ndvi_4wk_trend",
    "min_distance_km", "min_travel_hrs",
    "month", "is_wet_season",
    "flood_risk_lag1", "stress_x_distance",
    "price_pct_change",
]


def train_and_evaluate(panel):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, precision_recall_fscore_support, roc_curve)
    from sklearn.preprocessing import StandardScaler

    log.info("Training models on REAL data...")

    # Temporal split
    panel = panel.sort_values("date")
    split_date = panel["date"].quantile(0.8)
    train = panel[panel["date"] <= split_date]
    test = panel[panel["date"] > split_date]

    avail = [c for c in FEATURE_COLS if c in panel.columns]
    X_train = train[avail].fillna(0)
    y_train = train["spike"]
    X_test = test[avail].fillna(0)
    y_test = test["spike"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log.info(f"  Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()}) | spike rate: {y_train.mean()*100:.1f}%")
    log.info(f"  Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()}) | spike rate: {y_test.mean()*100:.1f}%")
    log.info(f"  Features: {len(avail)}")

    # Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average="binary", zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_prob) if y_test.nunique() > 1 else 0.5

    log.info(f"\n  Logistic Regression: P={lr_p:.3f} R={lr_r:.3f} F1={lr_f1:.3f} AUC={lr_auc:.3f}")

    coef_df = pd.DataFrame({"feature": avail, "coefficient": lr.coef_[0],
                             "abs_coef": np.abs(lr.coef_[0])}).sort_values("abs_coef", ascending=False)
    log.info(f"  Top coefficients:\n{coef_df.head(8).to_string(index=False)}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average="binary", zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_prob) if y_test.nunique() > 1 else 0.5

    log.info(f"  Random Forest:        P={rf_p:.3f} R={rf_r:.3f} F1={rf_f1:.3f} AUC={rf_auc:.3f}")

    fi_df = pd.DataFrame({"feature": avail, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    log.info(f"  Top features:\n{fi_df.head(8).to_string(index=False)}")

    # ── Save report ──
    report = []
    report.append("=" * 60)
    report.append("REAL DATA — PRICE SPIKE PREDICTION RESULTS")
    report.append("South Sumatra Province (PIHPS + GEE + CHIRPS)")
    report.append("=" * 60)
    report.append(f"\nDataset: {len(panel)} observations")
    report.append(f"Locations: {panel['district'].unique().tolist()}")
    report.append(f"Commodities: {panel['commodity'].unique().tolist()}")
    report.append(f"Period: {panel['date'].min().date()} to {panel['date'].max().date()}")
    report.append(f"Train: {len(train)} rows | Test: {len(test)} rows")
    report.append(f"Overall spike rate: {panel['spike'].mean()*100:.1f}%")
    report.append(f"\nFeatures ({len(avail)}): {avail}")
    report.append(f"\n{'─'*60}")
    report.append("LOGISTIC REGRESSION")
    report.append(f"Precision: {lr_p:.3f} | Recall: {lr_r:.3f} | F1: {lr_f1:.3f} | AUC: {lr_auc:.3f}")
    report.append(classification_report(y_test, lr_pred, target_names=["No spike", "Spike"], zero_division=0))
    report.append(f"Confusion Matrix:\n{confusion_matrix(y_test, lr_pred)}")
    report.append(f"\nCoefficients:\n{coef_df.to_string(index=False)}")
    report.append(f"\n{'─'*60}")
    report.append("RANDOM FOREST")
    report.append(f"Precision: {rf_p:.3f} | Recall: {rf_r:.3f} | F1: {rf_f1:.3f} | AUC: {rf_auc:.3f}")
    report.append(classification_report(y_test, rf_pred, target_names=["No spike", "Spike"], zero_division=0))
    report.append(f"Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
    report.append(f"\nFeature Importance:\n{fi_df.to_string(index=False)}")

    with open(OUTPUT_DIR / "real_model_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    fi_df.to_csv(OUTPUT_DIR / "real_feature_importance.csv", index=False)
    coef_df.to_csv(OUTPUT_DIR / "real_lr_coefficients.csv", index=False)

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (name, preds) in zip(axes, [("Logistic Regression", lr_pred), ("Random Forest", rf_pred)]):
            cm = confusion_matrix(y_test, preds)
            ax.imshow(cm, cmap="Blues")
            ax.set_title(name, fontsize=13, fontweight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["No spike","Spike"]); ax.set_yticklabels(["No spike","Spike"])
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                            color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=16)
        plt.suptitle("Real Data — South Sumatra Price Spike Prediction", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "real_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ROC
        fig, ax = plt.subplots(figsize=(7, 6))
        for name, prob, auc_val in [("Logistic Reg.", lr_prob, lr_auc), ("Random Forest", rf_prob, rf_auc)]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Real Data", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "real_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Feature importance
        fig, ax = plt.subplots(figsize=(8, 6))
        top = fi_df.head(12)
        colors = ["#2E75B6" if v > fi_df["importance"].median() else "#A0C4E8" for v in top["importance"]]
        ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
        ax.set_xlabel("Importance (Gini)")
        ax.set_title("Feature Importance — Real Data", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "real_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Price time series with spikes highlighted
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        for ax, commodity in zip(axes.flat, panel["commodity"].unique()):
            for district in panel["district"].unique():
                subset = panel[(panel["commodity"]==commodity) & (panel["district"]==district)]
                ax.plot(subset["date"], subset["price_rp_kg"], label=district, linewidth=0.8, alpha=0.8)
                spikes = subset[subset["spike"]==1]
                ax.scatter(spikes["date"], spikes["price_rp_kg"], color="red", s=15, zorder=5, alpha=0.6)
            ax.set_title(commodity, fontsize=11)
            ax.set_ylabel("Rp/kg")
            ax.legend(fontsize=7)
        plt.suptitle("Price Time Series with Detected Spikes (red dots)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "real_price_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close()

        log.info("  All plots saved to outputs/")
    except ImportError:
        log.warning("  matplotlib not available, skipping plots")

    # ── Risk map ──
    try:
        import folium
        latest_week = test["date"].max()
        latest = test[test["date"] == latest_week].copy()
        if len(latest) == 0:
            latest = test.tail(12)
        latest["spike_prob"] = rf.predict_proba(latest[avail].fillna(0))[:, 1]

        centroids = {
            "Kota Palembang": (-2.976, 104.775),
            "Kota Lubuk Linggau": (-3.300, 102.867),
            "Sumatera Selatan": (-3.3, 103.8),
        }

        m = folium.Map(location=[-3.2, 103.8], zoom_start=8, tiles="CartoDB positron")
        for _, row in latest.groupby("district")["spike_prob"].mean().reset_index().iterrows():
            coords = centroids.get(row["district"])
            if not coords: continue
            prob = row["spike_prob"]
            color = "#1a9850" if prob<0.2 else "#91cf60" if prob<0.4 else "#fee08b" if prob<0.6 else "#fc8d59" if prob<0.8 else "#d73027"
            folium.CircleMarker(
                location=coords, radius=15+prob*25,
                color=color, fill=True, fill_color=color, fill_opacity=0.7,
                popup=f"<b>{row['district']}</b><br>Spike risk: {prob:.0%}",
                tooltip=f"{row['district']}: {prob:.0%}"
            ).add_to(m)

        # Add producing district markers
        prod_centroids = {
            "Lahat": (-3.8, 103.55), "Kota Pagaralam": (-4.017, 103.267),
            "OKU Selatan": (-4.4, 104.1), "OKU Timur": (-3.75, 104.5),
            "OKU": (-4.05, 104.05),
        }
        for name, coords in prod_centroids.items():
            folium.CircleMarker(
                location=coords, radius=8,
                color="#2E75B6", fill=True, fill_color="#2E75B6", fill_opacity=0.5,
                tooltip=f"{name} (producing area)"
            ).add_to(m)

        m.save(str(OUTPUT_DIR / "real_risk_map.html"))
        log.info(f"  Risk map saved")
    except ImportError:
        log.warning("  folium not available")

    return {"lr_auc": lr_auc, "rf_auc": rf_auc, "lr_f1": lr_f1, "rf_f1": rf_f1}


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("HORTICULTURAL PRICE SPIKE EARLY-WARNING — REAL DATA")
    log.info("South Sumatra Province")
    log.info("=" * 60)

    prices, ndvi, rain = load_all()

    supply_ndvi = build_supply_ndvi(ndvi)
    local_ndvi = build_local_ndvi(ndvi)
    local_rain = build_local_rainfall(rain)

    panel = build_panel(prices, supply_ndvi, local_ndvi, local_rain)

    # Save panel
    panel_path = PROC_DIR / "real_panel_dataset.csv"
    panel.to_csv(panel_path, index=False)
    log.info(f"\nPanel dataset saved: {panel_path} ({len(panel)} rows x {len(panel.columns)} cols)")

    results = train_and_evaluate(panel)

    log.info(f"\n{'='*60}")
    log.info("PIPELINE COMPLETE — REAL DATA")
    log.info(f"{'='*60}")
    log.info(f"Logistic Regression: F1={results['lr_f1']:.3f} AUC={results['lr_auc']:.3f}")
    log.info(f"Random Forest:       F1={results['rf_f1']:.3f} AUC={results['rf_auc']:.3f}")
    log.info(f"\nOutputs in: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("real_*")):
        log.info(f"  {f.name}")


if __name__ == "__main__":
    main()
