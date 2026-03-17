"""
01_scrape_pihps.py — Scrape historical food commodity prices from PIHPS Nasional
================================================================================
Bank Indonesia's Pusat Informasi Harga Pangan Strategis (PIHPS)
URL: https://www.bi.go.id/hargapangan/

This script uses Playwright (headless browser) to:
1. Navigate to the PIHPS TabelHarga page
2. Select filters (commodity, province, date range)
3. Extract the rendered HTML table
4. Parse into a pandas DataFrame and save as CSV

SETUP (run once):
    pip install playwright pandas lxml
    playwright install chromium

USAGE:
    python 01_scrape_pihps.py

OUTPUT:
    data/raw/pihps_prices_raw.csv
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your needs
# ---------------------------------------------------------------------------

# Province to scrape
PROVINCE = "Sumatera Selatan"

# Commodities of interest (must match PIHPS dropdown text exactly)
COMMODITIES = [
    "Cabai Rawit Merah",
    "Cabai Merah Keriting",
    "Cabai Merah Besar",
    "Bawang Merah Ukuran Sedang",
]

# Date range — scrape in quarterly chunks to avoid timeouts
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# Chunk size in days (90 = quarterly, keeps table size manageable)
CHUNK_DAYS = 90

# Pause between requests (seconds) — be polite to BI servers
REQUEST_DELAY = 5

# Output paths
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "pihps_prices_raw.csv"

# Base URL
BASE_URL = "https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalKomoditas"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# APPROACH A: XHR INTERCEPTION (preferred — faster, cleaner)
# ---------------------------------------------------------------------------

def try_xhr_approach():
    """
    Attempt to intercept the XHR/fetch calls that PIHPS makes when loading
    table data. This is faster and more reliable than scraping rendered HTML.
    
    HOW TO FIND THE ACTUAL XHR ENDPOINT:
    1. Open Chrome DevTools (F12) on the PIHPS page
    2. Go to Network tab, filter by XHR/Fetch
    3. Select filters and click "Lihat Laporan"
    4. Look for the request that returns price data (usually JSON or HTML fragment)
    5. Right-click → Copy as cURL
    6. Update the URL, headers, and params below
    
    Returns None if the XHR approach can't be determined automatically.
    """
    log.info("Attempting XHR interception approach...")
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None
    
    captured_requests = []
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        # Intercept network requests to find the data endpoint
        def handle_response(response):
            url = response.url
            # Look for XHR calls that might contain price data
            if any(kw in url.lower() for kw in ['tabelharga', 'getharga', 'getdata', 'harga', 'price', 'report']):
                try:
                    body = response.text()
                    if len(body) > 500:  # likely contains data
                        captured_requests.append({
                            'url': url,
                            'status': response.status,
                            'method': response.request.method,
                            'headers': dict(response.request.headers),
                            'body_length': len(body),
                            'body_preview': body[:500]
                        })
                        log.info(f"  Captured XHR: {url[:80]}... ({len(body)} chars)")
                except Exception:
                    pass
        
        page.on("response", handle_response)
        
        log.info("  Loading PIHPS page...")
        page.goto(BASE_URL, timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Try to interact with the form to trigger data loading
        log.info("  Attempting to fill form and trigger data load...")
        try:
            # These selectors may need adjustment based on actual page structure
            # The exact selectors depend on the PIHPS frontend framework
            page.wait_for_selector("select, .select2, .dropdown", timeout=10000)
            time.sleep(1)
            
            # Take a screenshot for debugging
            screenshot_path = RAW_DIR / "pihps_page_screenshot.png"
            page.screenshot(path=str(screenshot_path))
            log.info(f"  Screenshot saved to {screenshot_path}")
            log.info("  Page title: " + page.title())
            
            # Dump all select elements for debugging
            selects = page.query_selector_all("select")
            log.info(f"  Found {len(selects)} <select> elements")
            for i, sel in enumerate(selects):
                sel_id = sel.get_attribute("id") or sel.get_attribute("name") or f"unnamed_{i}"
                options = sel.query_selector_all("option")
                opt_texts = [o.text_content().strip() for o in options[:5]]
                log.info(f"    Select #{i} (id={sel_id}): {opt_texts}...")
        except Exception as e:
            log.warning(f"  Form interaction failed: {e}")
        
        browser.close()
    
    if captured_requests:
        log.info(f"Captured {len(captured_requests)} potential data endpoints:")
        for req in captured_requests:
            log.info(f"  {req['method']} {req['url'][:100]}")
            log.info(f"    Body preview: {req['body_preview'][:200]}")
        
        # Save captured endpoints for manual inspection
        import json
        endpoints_file = RAW_DIR / "pihps_captured_endpoints.json"
        with open(endpoints_file, "w") as f:
            json.dump(captured_requests, f, indent=2, default=str)
        log.info(f"Captured endpoints saved to {endpoints_file}")
        log.info("NEXT STEP: Inspect the endpoints and update this script with the correct URL/params.")
        return captured_requests
    else:
        log.info("No XHR endpoints captured. Falling back to Approach B (HTML scraping).")
        return None


# ---------------------------------------------------------------------------
# APPROACH B: PLAYWRIGHT HTML TABLE SCRAPING (fallback — always works)
# ---------------------------------------------------------------------------

def generate_date_chunks(start_str, end_str, chunk_days=90):
    """Split date range into manageable chunks."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime("%d/%m/%Y"),  # PIHPS uses DD/MM/YYYY
            chunk_end.strftime("%d/%m/%Y"),
        ))
        current = chunk_end + timedelta(days=1)
    return chunks


def scrape_with_playwright():
    """
    Scrape PIHPS using Playwright browser automation.
    This is the reliable fallback that works regardless of the frontend framework.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)
    
    date_chunks = generate_date_chunks(START_DATE, END_DATE, CHUNK_DAYS)
    total_tasks = len(COMMODITIES) * len(date_chunks)
    all_data = []
    task_num = 0
    
    log.info(f"Starting scrape: {len(COMMODITIES)} commodities x {len(date_chunks)} date chunks = {total_tasks} tasks")
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()
        
        for commodity in COMMODITIES:
            for start_date, end_date in date_chunks:
                task_num += 1
                log.info(f"[{task_num}/{total_tasks}] {commodity} | {start_date} - {end_date}")
                
                try:
                    # Load the page fresh for each query (avoids stale state)
                    page.goto(BASE_URL, timeout=30000)
                    page.wait_for_load_state("networkidle")
                    time.sleep(2)
                    
                    # === FILL IN FORM FIELDS ===
                    # NOTE: The exact selectors below are TEMPLATES.
                    # You MUST inspect the actual PIHPS page and update these.
                    # Use the screenshot from try_xhr_approach() to identify selectors.
                    #
                    # Common patterns for Indonesian gov sites:
                    #   - Standard <select>: page.select_option("#komoditas", label=commodity)
                    #   - Select2 widget: page.click(".select2-komoditas"), then page.click(f"text={commodity}")
                    #   - Custom dropdown: page.click("#dd-komoditas"), page.click(f"li:has-text('{commodity}')")
                    
                    # --- Commodity selector ---
                    # TRY: page.select_option("select[name*='komoditas']", label=commodity)
                    # OR:  page.select_option("#komoditas", label=commodity)
                    try:
                        page.select_option("select:nth-of-type(1)", label=commodity, timeout=5000)
                    except Exception:
                        # If select_option fails, try clicking approach
                        log.warning(f"  select_option failed for commodity, trying click approach")
                        page.click("select:nth-of-type(1)")
                        time.sleep(0.5)
                        page.click(f"option:has-text('{commodity}')")
                    
                    time.sleep(1)
                    
                    # --- Province selector ---
                    try:
                        page.select_option("select:nth-of-type(2)", label=PROVINCE, timeout=5000)
                    except Exception:
                        log.warning(f"  select_option failed for province, trying click approach")
                        page.click("select:nth-of-type(2)")
                        time.sleep(0.5)
                        page.click(f"option:has-text('{PROVINCE}')")
                    
                    time.sleep(1)
                    
                    # --- Tick "Tampilkan Kab/Kota" if available ---
                    try:
                        page.check("input[type='checkbox']", timeout=3000)
                    except Exception:
                        pass  # checkbox may not exist
                    
                    # --- Report type: select "Harian" (daily) ---
                    try:
                        page.select_option("select:nth-of-type(4)", label="Harian", timeout=3000)
                    except Exception:
                        pass  # may not have this selector
                    
                    # --- Date fields ---
                    # These might be <input type="text"> with a datepicker
                    date_inputs = page.query_selector_all("input[type='text']")
                    if len(date_inputs) >= 2:
                        # Clear and fill start date
                        date_inputs[-2].click(click_count=3)
                        date_inputs[-2].fill(start_date)
                        time.sleep(0.5)
                        # Clear and fill end date
                        date_inputs[-1].click(click_count=3)
                        date_inputs[-1].fill(end_date)
                        time.sleep(0.5)
                    
                    # --- Click submit / "Lihat Laporan" ---
                    try:
                        page.click("button:has-text('Lihat'), input[type='submit'], a:has-text('Lihat')", timeout=5000)
                    except Exception:
                        # Try any button-like element
                        page.click("button, .btn", timeout=5000)
                    
                    # Wait for table to appear
                    time.sleep(3)
                    page.wait_for_selector("table", timeout=15000)
                    time.sleep(2)
                    
                    # === EXTRACT TABLE DATA ===
                    table_html = page.inner_html("table")
                    
                    # Parse with pandas
                    dfs = pd.read_html(f"<table>{table_html}</table>")
                    if dfs:
                        df = dfs[0]
                        df["commodity"] = commodity
                        df["scrape_start"] = start_date
                        df["scrape_end"] = end_date
                        df["province"] = PROVINCE
                        all_data.append(df)
                        log.info(f"  Extracted {len(df)} rows")
                    else:
                        log.warning(f"  No table data found")
                    
                except Exception as e:
                    log.error(f"  FAILED: {e}")
                    # Save error screenshot
                    err_path = RAW_DIR / f"error_{task_num}.png"
                    try:
                        page.screenshot(path=str(err_path))
                    except Exception:
                        pass
                
                # Polite delay
                time.sleep(REQUEST_DELAY)
        
        browser.close()
    
    return all_data


# ---------------------------------------------------------------------------
# DATA CLEANING (runs on whatever the scraper produces)
# ---------------------------------------------------------------------------

def clean_and_save(raw_dfs):
    """Merge all scraped DataFrames, clean columns, save to CSV."""
    if not raw_dfs:
        log.error("No data to clean. Check scraper output.")
        return None
    
    df = pd.concat(raw_dfs, ignore_index=True)
    log.info(f"Combined raw data: {len(df)} rows, {len(df.columns)} columns")
    log.info(f"Columns: {list(df.columns)}")
    
    # ---- Column renaming (adapt based on actual scraped columns) ----
    # PIHPS tables typically have columns like:
    #   No | Nama Pasar | Kab/Kota | Harga Kemarin | Harga Hari Ini | Perubahan
    # or for date-range reports:
    #   No | Kab/Kota | Tanggal1 | Tanggal2 | ... | TanggalN
    #
    # You'll need to inspect the actual output and adjust this mapping.
    # For now, save the raw data and you can clean it manually or ask me.
    
    # Save raw
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info(f"Raw data saved to {OUTPUT_FILE}")
    
    # Print sample
    log.info(f"\nSample data:\n{df.head().to_string()}")
    
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("PIHPS Price Data Scraper")
    log.info(f"Province: {PROVINCE}")
    log.info(f"Commodities: {COMMODITIES}")
    log.info(f"Date range: {START_DATE} to {END_DATE}")
    log.info("=" * 60)
    
    # Step 1: Try XHR interception first (discovery run)
    log.info("\n--- Phase 1: XHR Endpoint Discovery ---")
    xhr_result = try_xhr_approach()
    
    if xhr_result:
        log.info("\nXHR endpoints discovered! Check pihps_captured_endpoints.json")
        log.info("If the endpoint returns usable data, you can build a direct")
        log.info("requests-based scraper (much faster). Otherwise, continue to Phase 2.")
        
        response = input("\nProceed to Playwright scraping? (y/n): ").strip().lower()
        if response != "y":
            log.info("Exiting. Update script with discovered endpoints and re-run.")
            return
    
    # Step 2: Full Playwright scraping
    log.info("\n--- Phase 2: Playwright HTML Scraping ---")
    raw_data = scrape_with_playwright()
    
    # Step 3: Clean and save
    log.info("\n--- Phase 3: Cleaning & Saving ---")
    clean_and_save(raw_data)
    
    log.info("\nDone! Next step: run 04_generate_synthetic.py to test the pipeline,")
    log.info("or proceed directly to 05_modeling_pipeline.py with your real data.")


if __name__ == "__main__":
    main()
