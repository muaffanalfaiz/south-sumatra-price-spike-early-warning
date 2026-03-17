"""
01b_scrape_pihps_api.py — Direct API scraper for PIHPS
========================================================
Uses the discovered API endpoints from PIHPS to download price data
directly via HTTP requests. Much faster and more reliable than browser
automation.

Discovered endpoints:
  - GetRefCommodityAndCategory — list of commodities with IDs
  - GetRefProvince — list of provinces with IDs
  - GetGridDataKomoditas — the actual price data grid

USAGE:
    python scripts/01b_scrape_pihps_api.py

OUTPUT:
    data/raw/pihps_commodities.json     (commodity reference)
    data/raw/pihps_provinces.json       (province reference)
    data/raw/pihps_prices_raw.csv       (all scraped price data)
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ──
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.bi.go.id/hargapangan"

# Target commodities (we'll map these to IDs after fetching the reference)
TARGET_COMMODITIES = [
    "Cabai Rawit Merah",
    "Cabai Rawit Hijau",
    "Cabai Merah Besar",
    "Cabai Merah Keriting",
    "Bawang Merah",
    "Bawang Putih",
]

# Date range to scrape (in chunks)
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
CHUNK_DAYS = 30  # 1-month chunks to keep responses small

# Polite delay between API calls (seconds)
DELAY = 2

# ── Session setup ──
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE}/TabelHarga/PasarTradisionalKomoditas",
})


def fetch_page_first():
    """Load the main page first to establish cookies and session."""
    log.info("Loading PIHPS page to establish session...")
    resp = session.get(f"{BASE}/TabelHarga/PasarTradisionalKomoditas", timeout=60)
    log.info(f"  Page loaded: status {resp.status_code}, cookies: {list(session.cookies.keys())}")
    return resp.status_code == 200


def fetch_commodities():
    """Fetch the commodity reference list with IDs."""
    log.info("Fetching commodity list...")
    url = f"{BASE}/WebSite/TabelHarga/GetRefCommodityAndCategory"
    resp = session.get(url, params={"_": int(time.time() * 1000)}, timeout=30)

    if resp.status_code != 200:
        log.error(f"  Failed: HTTP {resp.status_code}")
        return None

    data = resp.json()
    # Save raw response
    with open(RAW_DIR / "pihps_commodities.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"  Got {len(data) if isinstance(data, list) else 'unknown'} items")
    log.info(f"  Saved to pihps_commodities.json")

    # Print the structure so we can identify commodity IDs
    if isinstance(data, list):
        for item in data[:30]:
            log.info(f"    {item}")
    elif isinstance(data, dict):
        for key in list(data.keys())[:10]:
            log.info(f"    {key}: {str(data[key])[:100]}")

    return data


def fetch_provinces():
    """Fetch the province reference list with IDs."""
    log.info("Fetching province list...")
    url = f"{BASE}/WebSite/TabelHarga/GetRefProvince"
    resp = session.get(url, params={"_": int(time.time() * 1000)}, timeout=30)

    if resp.status_code != 200:
        log.error(f"  Failed: HTTP {resp.status_code}")
        return None

    data = resp.json()
    with open(RAW_DIR / "pihps_provinces.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"  Got {len(data) if isinstance(data, list) else 'unknown'} items")
    log.info(f"  Saved to pihps_provinces.json")

    if isinstance(data, list):
        for item in data:
            log.info(f"    {item}")
    elif isinstance(data, dict):
        for key in list(data.keys())[:10]:
            log.info(f"    {key}: {str(data[key])[:100]}")

    return data


def fetch_grid_data(commodity_id=None, province_id=None,
                    start_date=None, end_date=None,
                    price_type="1",  # 1 = pasar tradisional
                    report_type="1",  # 1 = harian
                    show_kabkota="true"):
    """
    Fetch the price grid data from GetGridDataKomoditas.

    Parameters will be refined after we see the actual API parameter names
    from the first discovery call.
    """
    url = f"{BASE}/WebSite/TabelHarga/GetGridDataKomoditas"

    # These parameter names are educated guesses based on the URL pattern
    # captured in discovery. We'll refine after the first test call.
    params = {
        "_": int(time.time() * 1000),
    }

    # Try different parameter naming conventions
    if commodity_id:
        params["commodity_id"] = commodity_id
        params["commodityId"] = commodity_id
    if province_id:
        params["province_id"] = province_id
        params["provinceId"] = province_id
    if start_date:
        params["start_date"] = start_date
        params["startDate"] = start_date
        params["tgl_awal"] = start_date
    if end_date:
        params["end_date"] = end_date
        params["endDate"] = end_date
        params["tgl_akhir"] = end_date

    params["price_type_id"] = price_type
    params["priceTypeId"] = price_type
    params["report_type"] = report_type
    params["reportType"] = report_type
    params["showKabKota"] = show_kabkota

    resp = session.get(url, params=params, timeout=60)
    return resp


def discover_grid_params():
    """
    Make a test call to GetGridDataKomoditas to see what parameters
    it expects and what the response looks like.
    """
    log.info("\nDiscovering GetGridDataKomoditas parameters...")

    # First, try calling with no params to see default behavior
    url = f"{BASE}/WebSite/TabelHarga/GetGridDataKomoditas"
    resp = session.get(url, params={"_": int(time.time() * 1000)}, timeout=60)

    log.info(f"  Status: {resp.status_code}")
    log.info(f"  Content-Type: {resp.headers.get('content-type', 'unknown')}")
    log.info(f"  Response length: {len(resp.text)} chars")

    # Try to parse as JSON
    try:
        data = resp.json()
        log.info(f"  Response type: {type(data).__name__}")
        if isinstance(data, dict):
            log.info(f"  Keys: {list(data.keys())}")
            for key in list(data.keys())[:5]:
                val_str = str(data[key])[:200]
                log.info(f"    {key}: {val_str}")
        elif isinstance(data, list):
            log.info(f"  List length: {len(data)}")
            if data:
                log.info(f"  First item: {str(data[0])[:200]}")

        # Save for inspection
        with open(RAW_DIR / "pihps_grid_default.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved to pihps_grid_default.json")

        return data

    except Exception:
        # Maybe it's HTML
        log.info(f"  Not JSON. First 500 chars:\n{resp.text[:500]}")
        with open(RAW_DIR / "pihps_grid_default.html", "w", encoding="utf-8") as f:
            f.write(resp.text)
        log.info(f"  Saved to pihps_grid_default.html")
        return resp.text


def try_download_button():
    """
    Try to use the Download endpoint (the green button on the page).
    This might give us an Excel/CSV file directly.
    """
    log.info("\nTrying Download endpoint...")

    # Common patterns for download URLs
    download_urls = [
        f"{BASE}/WebSite/TabelHarga/DownloadKomoditas",
        f"{BASE}/WebSite/TabelHarga/Download",
        f"{BASE}/WebSite/TabelHarga/ExportKomoditas",
        f"{BASE}/WebSite/TabelHarga/Export",
    ]

    for url in download_urls:
        try:
            resp = session.get(url, params={"_": int(time.time() * 1000)}, timeout=30)
            ct = resp.headers.get("content-type", "")
            log.info(f"  {url.split('/')[-1]}: status={resp.status_code}, type={ct}, size={len(resp.content)}")

            if "excel" in ct or "spreadsheet" in ct or "octet-stream" in ct:
                # Got a file!
                ext = "xlsx" if "excel" in ct else "xls"
                filepath = RAW_DIR / f"pihps_download.{ext}"
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                log.info(f"  Downloaded file saved: {filepath}")
                return filepath
            elif resp.status_code == 200 and len(resp.content) > 1000:
                filepath = RAW_DIR / f"pihps_download_response.txt"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(resp.text[:5000])
                log.info(f"  Response saved for inspection: {filepath}")
        except Exception as e:
            log.info(f"  {url.split('/')[-1]}: failed ({e})")

    return None


def main():
    log.info("=" * 60)
    log.info("PIHPS API Scraper — Direct Endpoint Approach")
    log.info("=" * 60)

    # Step 1: Establish session
    if not fetch_page_first():
        log.error("Could not load PIHPS page. Check internet connection.")
        return

    time.sleep(DELAY)

    # Step 2: Fetch reference data (commodities and provinces)
    commodities = fetch_commodities()
    time.sleep(DELAY)
    provinces = fetch_provinces()
    time.sleep(DELAY)

    # Step 3: Discover the grid data endpoint parameters
    grid_data = discover_grid_params()
    time.sleep(DELAY)

    # Step 4: Try the download button
    download_result = try_download_button()

    # Step 5: Summary and next steps
    log.info("\n" + "=" * 60)
    log.info("DISCOVERY RESULTS")
    log.info("=" * 60)
    log.info(f"Commodities: {'Found' if commodities else 'Failed'} — check pihps_commodities.json")
    log.info(f"Provinces:   {'Found' if provinces else 'Failed'} — check pihps_provinces.json")
    log.info(f"Grid data:   Check pihps_grid_default.json or .html")
    log.info(f"Download:    {'Got file!' if download_result else 'No direct download found'}")
    log.info(f"\nAll files saved to: {RAW_DIR}")
    log.info(f"\nNEXT STEP: Send the JSON files back to Claude to build the final scraper")
    log.info(f"with exact parameter names and commodity/province IDs.")


if __name__ == "__main__":
    main()
