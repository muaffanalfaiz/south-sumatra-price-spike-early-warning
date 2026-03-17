# Satellite-Informed Horticultural Price Spike Early-Warning System
### South Sumatra, Indonesia

> **Can satellite signals anticipate food market stress before it reaches consumers?**  
> This project builds a working answer for South Sumatra's horticultural markets.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square)](https://scikit-learn.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-MODIS%20%7C%20CHIRPS-green?style=flat-square)](https://earthengine.google.com)
[![Manuscript](https://img.shields.io/badge/Manuscript-In%20Preparation-lightgrey?style=flat-square)]()

---

## The problem

Horticultural price spikes in South Sumatra — driven by harvest stress in producing districts like Lahat, Pagaralam, and OKU — hit urban consumers in Palembang and Lubuk Linggau within 2–4 weeks. By the time market monitoring systems detect a spike, it has already happened.

**This project asks:** can we detect the upstream conditions that precede a spike early enough to warn food-system decision-makers before prices move?

---

## What I built

A three-layer predictive pipeline that integrates:

| Layer | Data | Source |
|---|---|---|
| Market response | Weekly horticultural prices, spike labels | PIHPS (Pusat Informasi Harga Pangan Strategis) |
| Supply stress | District-level NDVI/EVI anomalies, 16-day composites | MODIS via Google Earth Engine |
| Friction & weather | Weekly rainfall, road accessibility, wet season flag | CHIRPS via GEE, OpenStreetMap |

**Commodities:** Cabai merah, cabai rawit, bawang merah, bawang putih  
**Markets monitored:** Palembang, Lubuk Linggau, South Sumatra aggregate  
**Dataset:** 1,860 observations across 155 weeks

The pipeline covers data acquisition, feature engineering with lagged indicators (2–4 week lags), binary spike classification, and decision-support risk outputs.

---

## Key findings

### Full model performance (PIHPS + NDVI + Rainfall + Accessibility)

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | **0.877** | 0.392 | 0.901 | 0.547 |
| Random Forest | **0.863** | 0.485 | 0.802 | 0.605 |

Logistic regression catches **73 out of 81 spike events** in the test set — missing only 8.  
Random forest produces fewer false alarms (69 vs 113), making it more suitable for operational alerting.

### Ablation experiment — what happens without price momentum?

The dominant feature in the full model is `price_pct_change` (importance: 0.472). Removing it entirely and retraining tests whether the **satellite + rainfall + seasonality** signals have genuine standalone early-warning value.

| Condition | AUC |
|---|---|
| Full model (with price momentum) | 0.877 |
| **Without price momentum** (satellite + weather + access only) | **0.762** |

AUC drops from 0.877 to 0.762 — meaningful, but the model remains well above chance. **Environmental signals alone carry real predictive signal 2–4 weeks before price spikes are visible in market data.**

### Commodity-specific finding

| Commodity group | Satellite-only AUC | Interpretation |
|---|---|---|
| Bawang merah | **0.767** | NDVI anomalies provide meaningful 2–4 week lead warning |
| Cabai merah/rawit | 0.487 | 16-day MODIS resolution is too coarse for rapid-onset chili shocks |

This is a genuine finding, not just a limitation. Cabai price shocks are driven by acute, fast-onset events (pest damage, sudden floods) that outpace satellite revisit frequency. Bawang merah, with its longer growing cycle and more gradual stress accumulation, is a better satellite early-warning target. **Higher temporal resolution data (Sentinel-2, 5-day revisit) may improve cabai prediction in future work.**

---

## Repository structure

```
south-sumatra-price-spike-early-warning/
│
├── README.md
│
├── data/
│   ├── raw/                          # Original PIHPS, GEE exports
│   └── processed/
│       ├── real_prices.csv           # Cleaned weekly price series
│       ├── real_panel_dataset.csv    # Full modeling panel with features
│       ├── real_model_ablation.csv   # Ablation experiment results
│       └── dashboard_ready/          # Flat tables for Power BI
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb     # PIHPS scraping, GEE export
│   ├── 02_feature_engineering.ipynb  # Lag features, spike labeling
│   ├── 03_modeling.ipynb             # LR + RF, evaluation, risk map
│   └── 04_ablation_experiment.ipynb  # Removing price momentum, commodity split
│
├── outputs/
│   ├── figures/                      # ROC curves, confusion matrices, timeseries
│   ├── model_results/                # Evaluation text files, feature importance CSVs
│   └── maps/                         # HTML risk map
│
├── powerbi/
│   ├── south_sumatra_dashboard.pbix  # Power BI dashboard file
│   └── screenshots/                  # Dashboard page previews
│
└── docs/
    └── index.md                      # GitHub Pages project site
```

---

## How to reproduce

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/south-sumatra-price-spike-early-warning.git
cd south-sumatra-price-spike-early-warning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook notebooks/01_data_acquisition.ipynb
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `geopandas`, `matplotlib`, `seaborn`, `requests`, `jupyter`

GEE exports require a registered Google Earth Engine account. Raw PIHPS data is scraped from the public API (no authentication required).

---

## Dashboard

An interactive Power BI dashboard covers four views:

- **Executive overview** — current spike risk by market, model KPIs
- **Price monitoring** — weekly price trends by commodity and location
- **Model drivers** — feature importance by signal type (market / NDVI / rainfall / access)
- **Model performance** — ROC curves, confusion matrices, ablation comparison

*Dashboard screenshots in `powerbi/screenshots/`. Full `.pbix` file included.*

---

## Data sources

| Source | Coverage | Access |
|---|---|---|
| [PIHPS](https://hargapangan.id) | Weekly horticultural prices, South Sumatra | Public web API |
| [MODIS MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1) | 16-day NDVI/EVI, 250m | Google Earth Engine (free account) |
| [CHIRPS](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY) | Daily rainfall | Google Earth Engine (free account) |
| [OpenStreetMap](https://www.openstreetmap.org) | Road network, accessibility | Open |
| BPS South Sumatra | District boundaries, administrative lookup | Public |

---

## Limitations

- Market coverage is limited to three monitored locations — results should not be generalised to all South Sumatra districts
- NDVI is a vegetation stress proxy, not a confirmed harvest measure
- Cabai early-warning from 16-day MODIS composites is insufficient — higher-resolution imagery is needed
- The model is predictive and mechanism-informed, not a clean causal identification

---

## Publication status

Manuscript in preparation.  
Target journal: *Computers and Electronics in Agriculture* (Scopus Q2, subscription route)  
Provisional title: *Predicting Horticultural Price Spikes in South Sumatra Using Market Prices, Satellite-Derived Vegetation Stress, Rainfall, and Machine Learning*

---

## About

Built by **Muaffan Alfaiz Wisaksono**  
MSc Agricultural Engineering (with High Distinction), Lincoln University, New Zealand  
LPDP Scholar | GIS Analyst | Precision Agriculture Researcher  
8× Scopus-indexed publications

[LinkedIn](https://www.linkedin.com/in/muaffanalfaiz) · [GitHub](https://github.com/muaffanalfaiz)

---

*This project is part of a portfolio demonstrating applied geospatial data science for agricultural and food-system intelligence.*
