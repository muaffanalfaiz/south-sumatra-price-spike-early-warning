"""
Dashboard CSV Export Script
Exports flat tables from processed data for Power BI
Run from the project root folder
"""

import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED = "data/processed"
DASHBOARD = "data/processed/dashboard_ready"
os.makedirs(DASHBOARD, exist_ok=True)

# ── Load source files ──────────────────────────────────────────────────────
print("Loading source files...")
prices     = pd.read_csv(f"{PROCESSED}/real_prices.csv")
panel      = pd.read_csv(f"{PROCESSED}/real_panel_dataset.csv")
ablation   = pd.read_csv(f"{PROCESSED}/real_model_ablation.csv")
feat_imp   = pd.read_csv("outputs/model_results/real_feature_importance.csv")
lr_coef    = pd.read_csv("outputs/model_results/real_lr_coefficients.csv")

print("Columns in prices:", prices.columns.tolist())
print("Columns in panel: ", panel.columns.tolist())
print("Columns in feat_imp:", feat_imp.columns.tolist())

# ── Table 1: prices_dashboard.csv ─────────────────────────────────────────
# Main fact table for price monitoring page
# Keep: date/week, commodity, location, price, spike flag
# Adjust column names below if they differ from what prints above

price_cols = prices.columns.tolist()

# Try to detect date, commodity, location, price, spike columns
date_col      = next((c for c in price_cols if 'date' in c.lower() or 'week' in c.lower()), price_cols[0])
commodity_col = next((c for c in price_cols if 'commodity' in c.lower() or 'komoditas' in c.lower() or 'nama' in c.lower()), None)
location_col  = next((c for c in price_cols if 'market' in c.lower() or 'location' in c.lower() or 'pasar' in c.lower() or 'kota' in c.lower()), None)
price_col     = next((c for c in price_cols if 'price' in c.lower() or 'harga' in c.lower()), None)
spike_col     = next((c for c in price_cols if 'spike' in c.lower()), None)

print(f"\nDetected columns → date: {date_col}, commodity: {commodity_col}, location: {location_col}, price: {price_col}, spike: {spike_col}")

keep = [c for c in [date_col, commodity_col, location_col, price_col, spike_col] if c]
prices_dash = prices[keep].copy()
prices_dash.columns = [c.lower().replace(' ', '_') for c in prices_dash.columns]
prices_dash.to_csv(f"{DASHBOARD}/prices_dashboard.csv", index=False)
print(f"✓ prices_dashboard.csv → {len(prices_dash)} rows")

# ── Table 2: model_scores.csv ─────────────────────────────────────────────
# Spike probability scores from the panel dataset
panel_cols = panel.columns.tolist()
score_candidates = [c for c in panel_cols if 'prob' in c.lower() or 'score' in c.lower() or 'predict' in c.lower() or 'spike' in c.lower()]
date_p    = next((c for c in panel_cols if 'date' in c.lower() or 'week' in c.lower()), panel_cols[0])
comm_p    = next((c for c in panel_cols if 'commodity' in c.lower() or 'komoditas' in c.lower()), None)
loc_p     = next((c for c in panel_cols if 'market' in c.lower() or 'location' in c.lower() or 'pasar' in c.lower()), None)

score_keep = [c for c in [date_p, comm_p, loc_p] + score_candidates if c]
score_keep = list(dict.fromkeys(score_keep))  # deduplicate
scores_dash = panel[score_keep].copy()
scores_dash.columns = [c.lower().replace(' ', '_') for c in scores_dash.columns]
scores_dash.to_csv(f"{DASHBOARD}/model_scores.csv", index=False)
print(f"✓ model_scores.csv → {len(scores_dash)} rows")

# ── Table 3: feature_importance.csv ───────────────────────────────────────
# Add signal category column for colour coding in Power BI
def categorise(feature):
    f = str(feature).lower()
    if 'price' in f or 'pct' in f or 'momentum' in f:
        return 'Price momentum'
    elif 'ndvi' in f or 'evi' in f or 'supply' in f or 'local' in f:
        return 'Satellite / NDVI'
    elif 'rain' in f or 'wet' in f or 'precip' in f:
        return 'Rainfall / weather'
    elif 'distance' in f or 'travel' in f or 'road' in f or 'access' in f or 'stress_x' in f:
        return 'Accessibility'
    elif 'month' in f or 'season' in f or 'year' in f:
        return 'Seasonality'
    else:
        return 'Other'

feat_imp['category'] = feat_imp.iloc[:, 0].apply(categorise)
feat_imp.to_csv(f"{DASHBOARD}/feature_importance.csv", index=False)
print(f"✓ feature_importance.csv → {len(feat_imp)} rows")

# ── Table 4: model_metrics.csv ────────────────────────────────────────────
metrics = pd.DataFrame([
    {"model": "Logistic Regression", "condition": "Full model",            "auc": 0.877, "precision": 0.392, "recall": 0.901, "f1": 0.547, "accuracy": 0.670},
    {"model": "Random Forest",       "condition": "Full model",            "auc": 0.863, "precision": 0.485, "recall": 0.802, "f1": 0.605, "accuracy": 0.770},
    {"model": "Logistic Regression", "condition": "Without price momentum","auc": 0.762, "precision": None,  "recall": None,  "f1": None,  "accuracy": None},
    {"model": "Bawang satellite-only","condition": "Commodity-specific",   "auc": 0.767, "precision": None,  "recall": None,  "f1": None,  "accuracy": None},
    {"model": "Cabai satellite-only", "condition": "Commodity-specific",   "auc": 0.487, "precision": None,  "recall": None,  "f1": None,  "accuracy": None},
])
metrics.to_csv(f"{DASHBOARD}/model_metrics.csv", index=False)
print(f"✓ model_metrics.csv → {len(metrics)} rows")

# ── Table 5: project_summary.csv ──────────────────────────────────────────
summary = pd.DataFrame([
    {"metric": "Total observations", "value": 1860},
    {"metric": "Monitored markets",  "value": 3},
    {"metric": "Commodities",        "value": 4},
    {"metric": "Weeks covered",      "value": 155},
    {"metric": "Spike rate (%)",     "value": 18.8},
    {"metric": "Best AUC",           "value": 0.877},
    {"metric": "Spikes caught (LR)", "value": 73},
    {"metric": "Spikes missed (LR)", "value": 8},
])
summary.to_csv(f"{DASHBOARD}/project_summary.csv", index=False)
print(f"✓ project_summary.csv → {len(summary)} rows")

print("\nAll done. Files saved to:", DASHBOARD)
print("\nNext: open Power BI Desktop and connect to the dashboard_ready folder.")
