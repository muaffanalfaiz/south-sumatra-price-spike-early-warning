// ======================================================================
// 02_gee_ndvi_export.js
// NDVI/EVI Harvest Stress Indicator — South Sumatra
// ======================================================================
// Paste this entire script into Google Earth Engine Code Editor:
//   https://code.earthengine.google.com/
//
// What it does:
//   1. Loads MODIS 16-day NDVI/EVI composites for South Sumatra
//   2. Builds a 5-year seasonal baseline (2018-2022)
//   3. Computes anomalies for the analysis period (2023-2025)
//   4. Aggregates anomalies to kabupaten/kota level
//   5. Exports a CSV to your Google Drive
//
// After export, the CSV goes into data/raw/sumsel_ndvi_anomaly.csv
// and feeds into 05_modeling_pipeline.py
// ======================================================================


// ──────────────────────────────────────────────────────────────────────
// 1. STUDY AREA: South Sumatra districts
// ──────────────────────────────────────────────────────────────────────

// GADM Level 2 gives kabupaten/kota boundaries
var districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Sumatera Selatan'));

// Province outline for clipping/display
var province = districts.geometry().dissolve();

Map.centerObject(province, 8);
Map.addLayer(province, {color: 'gray'}, 'South Sumatra', true, 0.3);
Map.addLayer(districts, {color: 'blue'}, 'Districts', true, 0.4);

// Print district names so you can verify coverage
print('Districts in South Sumatra:', districts.aggregate_array('ADM2_NAME').sort());
print('District count:', districts.size());


// ──────────────────────────────────────────────────────────────────────
// 2. LOAD MODIS VEGETATION INDEX DATA
// ──────────────────────────────────────────────────────────────────────

// MOD13Q1: 250m, 16-day composite, NDVI + EVI + quality flags
var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterBounds(province)
  .select(['NDVI', 'EVI', 'SummaryQA']);

// Quality filter: keep only good + marginal pixels
// SummaryQA: 0 = good, 1 = marginal, 2 = snow/ice, 3 = cloudy
function maskBadPixels(img) {
  var qa = img.select('SummaryQA');
  var good = qa.lte(1);  // 0 or 1
  return img.updateMask(good)
    .select(['NDVI', 'EVI'])
    .copyProperties(img, ['system:time_start']);
}

var modisClean = modis.map(maskBadPixels);


// ──────────────────────────────────────────────────────────────────────
// 3. BUILD SEASONAL BASELINE (5-year mean per 16-day window)
// ──────────────────────────────────────────────────────────────────────

var baselineStart = '2018-01-01';
var baselineEnd   = '2022-12-31';

var baseline = modisClean.filterDate(baselineStart, baselineEnd);

// Tag each image with its DOY window (0-22, 23 windows per year)
function addDoyWindow(img) {
  var doy = img.date().getRelative('day', 'year');
  var window = doy.divide(16).floor();
  return img.set('doy_window', window);
}

var baselineTagged = baseline.map(addDoyWindow);

// Compute mean NDVI/EVI for each of the 23 windows
var windowList = ee.List.sequence(0, 22);

var baselineMeans = ee.ImageCollection.fromImages(
  windowList.map(function(w) {
    var windowImgs = baselineTagged
      .filter(ee.Filter.eq('doy_window', w));
    var meanImg = windowImgs.mean().set('doy_window', w);
    // Preserve a time_start for sorting (use first image's date)
    var firstTime = windowImgs.aggregate_min('system:time_start');
    return meanImg.set('system:time_start', firstTime);
  })
);

print('Baseline windows computed:', baselineMeans.size());


// ──────────────────────────────────────────────────────────────────────
// 4. COMPUTE ANOMALIES FOR ANALYSIS PERIOD
// ──────────────────────────────────────────────────────────────────────

var analysisStart = '2023-01-01';
var analysisEnd   = '2025-12-31';

var analysis = modisClean
  .filterDate(analysisStart, analysisEnd)
  .map(addDoyWindow);

var analysisWithAnomaly = analysis.map(function(img) {
  var w = img.get('doy_window');
  
  // Get the matching baseline mean for this DOY window
  var baselineMean = baselineMeans
    .filter(ee.Filter.eq('doy_window', w))
    .first();
  
  // Anomaly = current - baseline mean
  var ndviAnomaly = img.select('NDVI')
    .subtract(baselineMean.select('NDVI'))
    .rename('NDVI_anomaly');
  
  var eviAnomaly = img.select('EVI')
    .subtract(baselineMean.select('EVI'))
    .rename('EVI_anomaly');
  
  return img
    .addBands(ndviAnomaly)
    .addBands(eviAnomaly)
    .set('doy_window', w);
});

print('Analysis images with anomaly:', analysisWithAnomaly.size());


// ──────────────────────────────────────────────────────────────────────
// 5. AGGREGATE TO DISTRICT LEVEL & EXPORT
// ──────────────────────────────────────────────────────────────────────

// For each 16-day composite, compute mean anomaly per district
var districtStats = analysisWithAnomaly.map(function(img) {
  var stats = img.select(['NDVI_anomaly', 'EVI_anomaly', 'NDVI', 'EVI'])
    .reduceRegions({
      collection: districts,
      reducer: ee.Reducer.mean(),
      scale: 250,
      crs: 'EPSG:4326'
    });
  
  // Add date info to each feature
  var dateStr = img.date().format('YYYY-MM-dd');
  return stats.map(function(f) {
    return f.set({
      'date': dateStr,
      'doy_window': img.get('doy_window')
    });
  });
});

// Flatten nested collection
var flatTable = districtStats.flatten();

print('Total district-date observations:', flatTable.size());
print('Sample row:', flatTable.first());

// Export to Google Drive as CSV
Export.table.toDrive({
  collection: flatTable,
  description: 'SumSel_NDVI_Anomaly_District_2023_2025',
  folder: 'GEE_Exports',
  fileNamePrefix: 'sumsel_ndvi_anomaly',
  fileFormat: 'CSV',
  selectors: [
    'ADM2_NAME',      // district name
    'date',            // YYYY-MM-dd
    'doy_window',      // 0-22
    'NDVI',            // raw NDVI (scaled x10000)
    'EVI',             // raw EVI (scaled x10000)
    'NDVI_anomaly',    // anomaly vs baseline (scaled x10000)
    'EVI_anomaly'      // anomaly vs baseline (scaled x10000)
  ]
});

print('Export task created. Go to Tasks tab (top right) and click RUN.');


// ──────────────────────────────────────────────────────────────────────
// 6. VISUALIZATION (optional — nice for screenshots)
// ──────────────────────────────────────────────────────────────────────

// Show the most recent anomaly map
var latestAnomaly = analysisWithAnomaly
  .select('NDVI_anomaly')
  .sort('system:time_start', false)
  .first()
  .clip(province);

Map.addLayer(latestAnomaly, {
  min: -2000,
  max: 2000,
  palette: ['d73027', 'fc8d59', 'fee08b', 'ffffbf', 'd9ef8b', '91cf60', '1a9850']
}, 'Latest NDVI anomaly');

// Show baseline NDVI for context
var baselineVis = baselineMeans.select('NDVI').mean().clip(province);
Map.addLayer(baselineVis, {
  min: 2000,
  max: 8000,
  palette: ['ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901', '66a000']
}, 'Baseline mean NDVI', false);


// ──────────────────────────────────────────────────────────────────────
// 7. BONUS: CHIRPS RAINFALL (for flood disruption proxy)
// ──────────────────────────────────────────────────────────────────────

// 7-day cumulative rainfall per district — export alongside NDVI
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(province)
  .filterDate(analysisStart, analysisEnd);

// Compute weekly cumulative rainfall
// Group by ISO week and sum
var weeks = ee.List.sequence(
  ee.Date(analysisStart).millis(),
  ee.Date(analysisEnd).millis(),
  7 * 24 * 60 * 60 * 1000  // 7 days in ms
);

var weeklyRainfall = ee.ImageCollection.fromImages(
  weeks.map(function(startMs) {
    var start = ee.Date(startMs);
    var end = start.advance(7, 'day');
    var weekSum = chirps.filterDate(start, end).sum();
    return weekSum.set({
      'system:time_start': startMs,
      'week_start': start.format('YYYY-MM-dd')
    });
  })
);

// Aggregate to district level
var rainfallByDistrict = weeklyRainfall.map(function(img) {
  var stats = img.reduceRegions({
    collection: districts,
    reducer: ee.Reducer.mean(),
    scale: 5000  // CHIRPS is ~5km resolution
  });
  return stats.map(function(f) {
    return f.set('week_start', img.get('week_start'));
  });
}).flatten();

Export.table.toDrive({
  collection: rainfallByDistrict,
  description: 'SumSel_Weekly_Rainfall_District_2023_2025',
  folder: 'GEE_Exports',
  fileNamePrefix: 'sumsel_weekly_rainfall',
  fileFormat: 'CSV',
  selectors: ['ADM2_NAME', 'week_start', 'mean']
});

print('Rainfall export task also created. Click RUN in Tasks tab.');
