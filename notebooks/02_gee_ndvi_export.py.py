"""
02_gee_ndvi_export_py.py — Python API version of the GEE NDVI export
=====================================================================
Use this if you prefer running from your local machine instead of
the GEE Code Editor. Identical logic, Python syntax.

SETUP:
    pip install earthengine-api
    earthengine authenticate

USAGE:
    python 02_gee_ndvi_export_py.py

OUTPUT:
    Exports CSV to your Google Drive folder 'GEE_Exports'
    Download it and place in data/raw/sumsel_ndvi_anomaly.csv
"""

import ee
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Authenticate & initialize ──
ee.Authenticate()
ee.Initialize(project="your-gee-project-id")  # CHANGE THIS to your GEE project

# ── Config ──
BASELINE_START = "2018-01-01"
BASELINE_END = "2022-12-31"
ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2025-12-31"
EXPORT_FOLDER = "GEE_Exports"

log.info("Loading South Sumatra districts from GADM...")
districts = (ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq("ADM1_NAME", "Sumatera Selatan")))
province = districts.geometry().dissolve()

log.info(f"Districts found: {districts.size().getInfo()}")
log.info(f"Names: {districts.aggregate_array('ADM2_NAME').sort().getInfo()}")

# ── Load & clean MODIS ──
def mask_bad_pixels(img):
    qa = img.select("SummaryQA")
    good = qa.lte(1)
    return (img.updateMask(good)
        .select(["NDVI", "EVI"])
        .copyProperties(img, ["system:time_start"]))

modis = (ee.ImageCollection("MODIS/061/MOD13Q1")
    .filterBounds(province)
    .select(["NDVI", "EVI", "SummaryQA"])
    .map(mask_bad_pixels))

# ── Tag with DOY window ──
def add_doy_window(img):
    doy = img.date().getRelative("day", "year")
    window = doy.divide(16).floor()
    return img.set("doy_window", window)

# ── Build baseline ──
log.info("Computing 5-year seasonal baseline...")
baseline = modis.filterDate(BASELINE_START, BASELINE_END).map(add_doy_window)

def compute_window_mean(w):
    w = ee.Number(w)
    window_imgs = baseline.filter(ee.Filter.eq("doy_window", w))
    return (window_imgs.mean()
        .set("doy_window", w)
        .set("system:time_start", window_imgs.aggregate_min("system:time_start")))

baseline_means = ee.ImageCollection.fromImages(
    ee.List.sequence(0, 22).map(compute_window_mean)
)

# ── Compute anomalies ──
log.info("Computing anomalies for analysis period...")
analysis = modis.filterDate(ANALYSIS_START, ANALYSIS_END).map(add_doy_window)

def compute_anomaly(img):
    w = img.get("doy_window")
    baseline_mean = baseline_means.filter(ee.Filter.eq("doy_window", w)).first()
    ndvi_anom = img.select("NDVI").subtract(baseline_mean.select("NDVI")).rename("NDVI_anomaly")
    evi_anom = img.select("EVI").subtract(baseline_mean.select("EVI")).rename("EVI_anomaly")
    return img.addBands(ndvi_anom).addBands(evi_anom)

analysis_anomaly = analysis.map(compute_anomaly)

# ── Aggregate to districts ──
log.info("Aggregating to district level...")

def reduce_to_districts(img):
    stats = (img.select(["NDVI_anomaly", "EVI_anomaly", "NDVI", "EVI"])
        .reduceRegions(
            collection=districts,
            reducer=ee.Reducer.mean(),
            scale=250,
            crs="EPSG:4326"
        ))
    date_str = img.date().format("YYYY-MM-dd")
    return stats.map(lambda f: f.set({
        "date": date_str,
        "doy_window": img.get("doy_window")
    }))

district_stats = analysis_anomaly.map(reduce_to_districts).flatten()

# ── Export NDVI ──
log.info("Starting NDVI export task...")
ndvi_task = ee.batch.Export.table.toDrive(
    collection=district_stats,
    description="SumSel_NDVI_Anomaly_District",
    folder=EXPORT_FOLDER,
    fileNamePrefix="sumsel_ndvi_anomaly",
    fileFormat="CSV",
    selectors=["ADM2_NAME", "date", "doy_window", "NDVI", "EVI", "NDVI_anomaly", "EVI_anomaly"]
)
ndvi_task.start()
log.info(f"NDVI export task started: {ndvi_task.status()}")

# ── Export CHIRPS rainfall ──
log.info("Computing weekly rainfall...")
chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterBounds(province)
    .filterDate(ANALYSIS_START, ANALYSIS_END))

week_starts = ee.List.sequence(
    ee.Date(ANALYSIS_START).millis(),
    ee.Date(ANALYSIS_END).millis(),
    7 * 24 * 60 * 60 * 1000
)

def weekly_sum(start_ms):
    start = ee.Date(start_ms)
    end = start.advance(7, "day")
    return (chirps.filterDate(start, end).sum()
        .set("system:time_start", start_ms)
        .set("week_start", start.format("YYYY-MM-dd")))

weekly_rainfall = ee.ImageCollection.fromImages(week_starts.map(weekly_sum))

def reduce_rainfall(img):
    stats = img.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=5000
    )
    return stats.map(lambda f: f.set("week_start", img.get("week_start")))

rainfall_stats = weekly_rainfall.map(reduce_rainfall).flatten()

rain_task = ee.batch.Export.table.toDrive(
    collection=rainfall_stats,
    description="SumSel_Weekly_Rainfall_District",
    folder=EXPORT_FOLDER,
    fileNamePrefix="sumsel_weekly_rainfall",
    fileFormat="CSV",
    selectors=["ADM2_NAME", "week_start", "mean"]
)
rain_task.start()
log.info(f"Rainfall export task started: {rain_task.status()}")

log.info("Both export tasks submitted. Check GEE Tasks panel or run:")
log.info("  import ee; print(ee.batch.Task.list())")
log.info("Download CSVs from Google Drive when complete.")
