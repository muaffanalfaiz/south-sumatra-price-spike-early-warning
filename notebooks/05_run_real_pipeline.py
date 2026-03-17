"""
05_modeling_pipeline.py — End-to-end price spike prediction pipeline
=====================================================================
Merges all data layers, engineers lagged features, trains models,
and evaluates prediction performance.

USAGE:
    # Test with synthetic data:
    python 05_modeling_pipeline.py --synthetic
    
    # Run with real data (after scraping):
    python 05_modeling_pipeline.py

OUTPUT:
    data/processed/model_panel_dataset.csv
    outputs/model_evaluation_report.txt
    outputs/feature_importance.csv
    outputs/confusion_matrix.png
    outputs/roc_curve.png
    outputs/risk_map_latest.html (Folium interactive map)
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 1. DATA LOADING
# ======================================================================

def load_data(synthetic=False):
    """Load all four data sources."""
    prefix = "synthetic_" if synthetic else ""
    log.info(f"Loading {'synthetic' if synthetic else 'real'} data...")

    # Prices
    prices_path = PROC_DIR / f"{prefix}prices.csv"
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    log.info(f"  Prices: {len(prices)} rows, {prices['district'].nunique()} districts, {prices['commodity'].nunique()} commodities")

    # NDVI
    ndvi_path = PROC_DIR / f"{prefix}ndvi.csv"
    ndvi = pd.read_csv(ndvi_path, parse_dates=["date"])
    ndvi = ndvi.rename(columns={"ADM2_NAME": "district"})
    # Convert MODIS scale to actual NDVI
    for col in ["NDVI", "EVI", "NDVI_anomaly", "EVI_anomaly"]:
        if col in ndvi.columns:
            ndvi[col] = ndvi[col] / 10000.0
    log.info(f"  NDVI: {len(ndvi)} rows")

    # Rainfall
    rain_path = PROC_DIR / f"{prefix}rainfall.csv"
    rainfall = pd.read_csv(rain_path, parse_dates=["week_start"])
    rainfall = rainfall.rename(columns={"ADM2_NAME": "district", "mean": "rainfall_mm"})
    log.info(f"  Rainfall: {len(rainfall)} rows")

    # Road distances
    dist_path = PROC_DIR / f"{prefix}road_distances.csv"
    distances = pd.read_csv(dist_path)
    log.info(f"  Distances: {len(distances)} rows")

    return prices, ndvi, rainfall, distances


# ======================================================================
# 2. FEATURE ENGINEERING
# ======================================================================

def resample_ndvi_to_weekly(ndvi):
    """
    MODIS is 16-day composite. Resample to weekly by forward-filling
    the nearest 16-day value for each district.
    """
    log.info("  Resampling NDVI from 16-day to weekly...")
    
    weekly_frames = []
    for district in ndvi["district"].unique():
        d = ndvi[ndvi["district"] == district].set_index("date").sort_index()
        # Resample to weekly, forward-fill
        d_weekly = d.resample("W-MON").ffill()
        d_weekly["district"] = district
        weekly_frames.append(d_weekly.reset_index())
    
    return pd.concat(weekly_frames, ignore_index=True)


def build_panel(prices, ndvi, rainfall, distances):
    """
    Merge all data into a single panel dataset.
    Each row = (district, week, commodity) observation.
    """
    log.info("Building panel dataset...")

    # 1. Resample NDVI to weekly
    ndvi_weekly = resample_ndvi_to_weekly(ndvi)

    # 2. Merge prices + NDVI (on district + week)
    panel = prices.merge(
        ndvi_weekly[["date", "district", "NDVI_anomaly", "EVI_anomaly", "NDVI"]],
        on=["date", "district"],
        how="left"
    )

    # 3. Merge rainfall (on district + week)
    # Match rainfall week_start to price date (both should be Monday-aligned)
    rainfall["date"] = rainfall["week_start"]
    panel = panel.merge(
        rainfall[["date", "district", "rainfall_mm"]],
        on=["date", "district"],
        how="left"
    )

    # 4. Merge road distances (static, on district only)
    panel = panel.merge(
        distances[["district", "min_distance_km", "min_travel_hrs"]],
        on="district",
        how="left"
    )

    # 5. Create lagged features
    log.info("  Creating lagged features...")
    panel = panel.sort_values(["district", "commodity", "date"])

    for lag in [1, 2, 3, 4]:
        panel[f"ndvi_anom_lag{lag}"] = (
            panel.groupby(["district", "commodity"])["NDVI_anomaly"]
            .shift(lag)
        )
        panel[f"rain_lag{lag}"] = (
            panel.groupby(["district", "commodity"])["rainfall_mm"]
            .shift(lag)
        )

    # 6. Rolling features
    panel["rain_4wk_sum"] = (
        panel.groupby(["district", "commodity"])["rainfall_mm"]
        .transform(lambda x: x.rolling(4, min_periods=2).sum())
    )
    panel["ndvi_4wk_trend"] = (
        panel.groupby(["district", "commodity"])["NDVI_anomaly"]
        .transform(lambda x: x.rolling(4, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0
        ))
    )

    # 7. Calendar features
    panel["month"] = panel["date"].dt.month
    panel["is_wet_season"] = panel["month"].isin([11, 12, 1, 2, 3]).astype(int)

    # 8. Flood disruption flag
    # Heavy rainfall (>100mm/week) along supply routes suggests disruption
    panel["flood_risk"] = (panel["rainfall_mm"] > 100).astype(int)
    panel["flood_risk_lag1"] = panel.groupby(["district", "commodity"])["flood_risk"].shift(1)

    # 9. Interaction feature: NDVI stress + road distance
    panel["stress_x_distance"] = panel["ndvi_anom_lag2"] * panel["min_distance_km"]

    # Drop rows with NaN from lagging
    before = len(panel)
    panel = panel.dropna(subset=["ndvi_anom_lag2", "rain_lag1", "spike"])
    log.info(f"  Panel: {before} -> {len(panel)} rows (dropped {before - len(panel)} NaN lag rows)")

    return panel


# ======================================================================
# 3. MODELING
# ======================================================================

FEATURE_COLS = [
    "ndvi_anom_lag2", "ndvi_anom_lag3", "ndvi_anom_lag4",
    "rain_lag1", "rain_lag2",
    "rain_4wk_sum", "ndvi_4wk_trend",
    "min_distance_km", "min_travel_hrs",
    "month", "is_wet_season",
    "flood_risk_lag1",
    "stress_x_distance",
]


def train_and_evaluate(panel):
    """Train logistic regression + Random Forest, evaluate on temporal test set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
        precision_recall_fscore_support, roc_curve
    )
    from sklearn.preprocessing import StandardScaler

    log.info("Training models...")

    # Temporal split: train on first 80%, test on last 20%
    panel = panel.sort_values("date")
    split_date = panel["date"].quantile(0.8)
    train = panel[panel["date"] <= split_date]
    test = panel[panel["date"] > split_date]

    log.info(f"  Train: {len(train)} rows ({train['date'].min()} to {train['date'].max()})")
    log.info(f"  Test:  {len(test)} rows ({test['date'].min()} to {test['date'].max()})")
    log.info(f"  Train spike rate: {train['spike'].mean()*100:.1f}%")
    log.info(f"  Test spike rate:  {test['spike'].mean()*100:.1f}%")

    # Features
    avail_features = [c for c in FEATURE_COLS if c in panel.columns]
    X_train = train[avail_features].fillna(0)
    y_train = train["spike"]
    X_test = test[avail_features].fillna(0)
    y_test = test["spike"]

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ── Model A: Logistic Regression ──
    log.info("\n  Model A: Logistic Regression")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average="binary", zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_prob) if y_test.nunique() > 1 else 0.5
    log.info(f"    Precision: {lr_p:.3f} | Recall: {lr_r:.3f} | F1: {lr_f1:.3f} | AUC: {lr_auc:.3f}")

    # Coefficients
    coef_df = pd.DataFrame({
        "feature": avail_features,
        "coefficient": lr.coef_[0],
        "abs_coef": np.abs(lr.coef_[0])
    }).sort_values("abs_coef", ascending=False)
    log.info(f"    Top features:\n{coef_df.head(8).to_string(index=False)}")

    results["logistic_regression"] = {
        "precision": lr_p, "recall": lr_r, "f1": lr_f1, "auc": lr_auc,
        "model": lr, "predictions": lr_pred, "probabilities": lr_prob,
        "coefficients": coef_df,
    }

    # ── Model B: Random Forest ──
    log.info("\n  Model B: Random Forest")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average="binary", zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_prob) if y_test.nunique() > 1 else 0.5
    log.info(f"    Precision: {rf_p:.3f} | Recall: {rf_r:.3f} | F1: {rf_f1:.3f} | AUC: {rf_auc:.3f}")

    # Feature importance
    fi_df = pd.DataFrame({
        "feature": avail_features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    log.info(f"    Top features:\n{fi_df.head(8).to_string(index=False)}")

    results["random_forest"] = {
        "precision": rf_p, "recall": rf_r, "f1": rf_f1, "auc": rf_auc,
        "model": rf, "predictions": rf_pred, "probabilities": rf_prob,
        "feature_importance": fi_df,
    }

    # ── Save outputs ──
    
    # Classification reports
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("PRICE SPIKE PREDICTION — MODEL EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"\nDataset: {len(panel)} total observations")
    report_lines.append(f"Train period: {train['date'].min().date()} to {train['date'].max().date()}")
    report_lines.append(f"Test period:  {test['date'].min().date()} to {test['date'].max().date()}")
    report_lines.append(f"Overall spike rate: {panel['spike'].mean()*100:.1f}%")
    report_lines.append(f"\nFeatures used: {avail_features}")
    report_lines.append(f"\n{'─'*60}")
    report_lines.append("LOGISTIC REGRESSION")
    report_lines.append(f"{'─'*60}")
    report_lines.append(f"Precision: {lr_p:.3f} | Recall: {lr_r:.3f} | F1: {lr_f1:.3f} | AUC: {lr_auc:.3f}")
    report_lines.append("\nClassification Report:")
    report_lines.append(classification_report(y_test, lr_pred, target_names=["No spike", "Spike"], zero_division=0))
    report_lines.append(f"\nConfusion Matrix:\n{confusion_matrix(y_test, lr_pred)}")
    report_lines.append(f"\nCoefficients:\n{coef_df.to_string(index=False)}")
    report_lines.append(f"\n{'─'*60}")
    report_lines.append("RANDOM FOREST")
    report_lines.append(f"{'─'*60}")
    report_lines.append(f"Precision: {rf_p:.3f} | Recall: {rf_r:.3f} | F1: {rf_f1:.3f} | AUC: {rf_auc:.3f}")
    report_lines.append("\nClassification Report:")
    report_lines.append(classification_report(y_test, rf_pred, target_names=["No spike", "Spike"], zero_division=0))
    report_lines.append(f"\nConfusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
    report_lines.append(f"\nFeature Importance:\n{fi_df.to_string(index=False)}")

    report_path = OUTPUT_DIR / "model_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    log.info(f"\nReport saved to {report_path}")

    # Feature importance CSV
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # ── Generate plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Confusion matrix heatmap
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (name, res) in zip(axes, results.items()):
            cm = confusion_matrix(y_test, res["predictions"])
            ax.imshow(cm, cmap="Blues", interpolation="nearest")
            ax.set_title(name.replace("_", " ").title())
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No spike", "Spike"])
            ax.set_yticklabels(["No spike", "Spike"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

        # ROC curve
        fig, ax = plt.subplots(figsize=(7, 6))
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res["probabilities"])
            ax.plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC={res["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Price Spike Prediction")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"ROC curve saved to {OUTPUT_DIR / 'roc_curve.png'}")

        # Feature importance bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        top_fi = fi_df.head(10)
        ax.barh(top_fi["feature"][::-1], top_fi["importance"][::-1], color="#2E75B6")
        ax.set_xlabel("Importance (Gini)")
        ax.set_title("Random Forest — Top 10 Feature Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Feature importance chart saved")

    except ImportError:
        log.warning("matplotlib not available, skipping plots")

    # ── Risk map (latest week prediction) ──
    try:
        generate_risk_map(test, rf, avail_features)
    except Exception as e:
        log.warning(f"Risk map generation skipped: {e}")

    return results, test


# ======================================================================
# 4. RISK MAP
# ======================================================================

def generate_risk_map(test_data, model, feature_cols):
    """Generate an interactive Folium map showing spike risk by district."""
    try:
        import folium
    except ImportError:
        log.warning("folium not installed, skipping risk map. Run: pip install folium")
        return

    log.info("Generating interactive risk map...")

    # Get latest week predictions
    latest_week = test_data["date"].max()
    latest = test_data[test_data["date"] == latest_week].copy()

    if latest.empty:
        log.warning("No data for latest week, using last available week")
        latest = test_data.tail(len(test_data["district"].unique()) * len(test_data["commodity"].unique()))

    # Average spike probability across commodities per district
    X_latest = latest[feature_cols].fillna(0)
    latest["spike_prob"] = model.predict_proba(X_latest)[:, 1]

    district_risk = (latest.groupby("district")["spike_prob"]
        .mean().reset_index()
        .rename(columns={"spike_prob": "avg_spike_prob"}))

    # District centroids (from script 03)
    centroids = {
        "Palembang": (-2.976, 104.775), "Ogan Komering Ilir": (-3.200, 105.400),
        "Ogan Komering Ulu": (-4.050, 104.050), "Muara Enim": (-3.700, 103.750),
        "Lahat": (-3.800, 103.550), "Musi Rawas": (-3.100, 102.900),
        "Musi Banyuasin": (-2.700, 104.200), "Banyuasin": (-2.500, 104.800),
        "OKU Selatan": (-4.400, 104.100), "OKU Timur": (-3.750, 104.500),
        "Ogan Ilir": (-3.250, 104.650), "Empat Lawang": (-3.950, 103.250),
        "PALI": (-3.350, 103.800), "Musi Rawas Utara": (-2.800, 102.700),
        "Lubuklinggau": (-3.300, 102.867), "Prabumulih": (-3.433, 104.233),
        "Pagar Alam": (-4.017, 103.267),
    }

    # Create map
    m = folium.Map(location=[-3.3, 103.8], zoom_start=8, tiles="CartoDB positron")

    for _, row in district_risk.iterrows():
        name = row["district"]
        prob = row["avg_spike_prob"]
        coords = centroids.get(name)
        if coords is None:
            continue

        # Color: green (low risk) to red (high risk)
        if prob < 0.2:
            color = "#1a9850"
        elif prob < 0.4:
            color = "#91cf60"
        elif prob < 0.6:
            color = "#fee08b"
        elif prob < 0.8:
            color = "#fc8d59"
        else:
            color = "#d73027"

        folium.CircleMarker(
            location=coords,
            radius=12 + prob * 20,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{name}</b><br>Spike probability: {prob:.1%}",
            tooltip=f"{name}: {prob:.0%} risk"
        ).add_to(m)

    # Title
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:999;
         background:white;padding:8px 16px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.2);
         font-family:Arial;font-size:14px;">
        <b>Horticultural Price Spike Risk Map</b> — South Sumatra | Week of {latest_week.date()}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    map_path = OUTPUT_DIR / "risk_map_latest.html"
    m.save(str(map_path))
    log.info(f"Risk map saved to {map_path}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Price spike prediction pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HORTICULTURAL PRICE SPIKE EARLY-WARNING SYSTEM")
    log.info("South Sumatra Province")
    log.info("=" * 60)

    # Load data
    prices, ndvi, rainfall, distances = load_data(synthetic=args.synthetic)

    # Build panel dataset
    panel = build_panel(prices, ndvi, rainfall, distances)

    # Save panel for inspection
    panel_path = PROC_DIR / "model_panel_dataset.csv"
    panel.to_csv(panel_path, index=False)
    log.info(f"Panel dataset saved to {panel_path} ({len(panel)} rows, {len(panel.columns)} columns)")

    # Train and evaluate
    results, test_data = train_and_evaluate(panel)

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"Check outputs in: {OUTPUT_DIR}")
    log.info("Files generated:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        log.info(f"  {f.name}")


if __name__ == "__main__":
    main()
