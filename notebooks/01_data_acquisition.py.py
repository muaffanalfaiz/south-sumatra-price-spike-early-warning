"""
01c_scrape_pihps_final.py — Production PIHPS price scraper
============================================================
Uses the exact API endpoints and parameter names discovered from
the PIHPS website. No browser automation needed.

TESTED ENDPOINT:
  GET /WebSite/TabelHarga/GetGridDataKomoditas
  ?price_type_id=1
  &comcat_id=cat_7
  &province_id=8
  &regency_id=
  &showKota=true
  &showPasar=false
  &tipe_laporan=1
  &start_date=2023-01-01
  &end_date=2023-01-31

USAGE:
    python scripts/01c_scrape_pihps_final.py

OUTPUT:
    data/raw/pihps_prices_all.csv   (complete dataset)
    data/raw/pihps_chunks/          (individual API responses as JSON)
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNKS_DIR = RAW_DIR / "pihps_chunks"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.bi.go.id/hargapangan"
GRID_URL = f"{BASE}/WebSite/TabelHarga/GetGridDataKomoditas"

# ── Exact parameter values from discovery ──

# Province: Sumatera Selatan = 8
PROVINCE_ID = 8
PROVINCE_NAME = "Sumatera Selatan"

# Categories to scrape (each category returns all its child commodities)
# cat_7 = Cabai Merah (contains: Cabai Merah Besar, Cabai Merah Keriting)
# cat_8 = Cabai Rawit (contains: Cabai Rawit Hijau, Cabai Rawit Merah)
# cat_5 = Bawang Merah (contains: Bawang Merah Ukuran Sedang)
# cat_6 = Bawang Putih (contains: Bawang Putih Ukuran Sedang)
CATEGORIES = {
    "cat_5": "Bawang Merah",
    "cat_6": "Bawang Putih",
    "cat_7": "Cabai Merah",
    "cat_8": "Cabai Rawit",
}

# Date range
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# Chunk size in days (PIHPS shows ~8 day columns per page by default,
# but the API should accept longer ranges. Start with 30-day chunks.)
CHUNK_DAYS = 30

# Price type: 1 = pasar tradisional
PRICE_TYPE_ID = 1

# Report type: 1 = harian (daily)
TIPE_LAPORAN = 1

# Show kabupaten/kota breakdown
SHOW_KOTA = "true"
SHOW_PASAR = "false"

# Polite delay between API calls
DELAY = 3

# ═══════════════════════════════════════════════════════════
# SESSION SETUP
# ═══════════════════════════════════════════════════════════

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE}/TabelHarga/PasarTradisionalKomoditas",
})


def establish_session():
    """Load the main page to get cookies."""
    log.info("Establishing session with PIHPS...")
    resp = session.get(f"{BASE}/TabelHarga/PasarTradisionalKomoditas", timeout=60)
    if resp.status_code == 200:
        cookies = list(session.cookies.keys())
        log.info(f"  Session established. Cookies: {cookies}")
        return True
    else:
        log.error(f"  Failed: HTTP {resp.status_code}")
        return False


def generate_date_chunks(start_str, end_str, chunk_days=30):
    """Split date range into chunks."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_grid(comcat_id, start_date, end_date):
    """
    Fetch price grid data for one category and date range.
    Returns the parsed JSON response or None on failure.
    """
    params = {
        "price_type_id": PRICE_TYPE_ID,
        "comcat_id": comcat_id,
        "province_id": PROVINCE_ID,
        "regency_id": "",
        "showKota": SHOW_KOTA,
        "showPasar": SHOW_PASAR,
        "tipe_laporan": TIPE_LAPORAN,
        "start_date": start_date,
        "end_date": end_date,
        "_": int(time.time() * 1000),
    }

    try:
        resp = session.get(GRID_URL, params=params, timeout=60)

        if resp.status_code != 200:
            log.warning(f"    HTTP {resp.status_code}")
            return None

        ct = resp.headers.get("content-type", "")
        if "json" not in ct:
            log.warning(f"    Got {ct} instead of JSON ({len(resp.text)} chars)")
            # Save for debugging
            debug_path = CHUNKS_DIR / f"debug_{comcat_id}_{start_date}.html"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(resp.text[:5000])
            return None

        data = resp.json()
        return data

    except requests.exceptions.Timeout:
        log.warning(f"    Timeout")
        return None
    except Exception as e:
        log.warning(f"    Error: {e}")
        return None


def parse_grid_response(data, comcat_id, cat_name, start_date, end_date):
    """
    Parse the JSON grid response into rows.
    The exact structure will depend on what the API returns.
    We handle multiple possible formats.
    """
    rows = []

    if data is None:
        return rows

    # The response could be:
    # A) {"data": [...]} — list of row objects
    # B) {"columns": [...], "rows": [...]} — tabular
    # C) A list directly [...]
    # D) Something else

    actual_data = data
    if isinstance(data, dict):
        if "data" in data:
            actual_data = data["data"]
        elif "rows" in data:
            actual_data = data["rows"]
        elif "Data" in data:
            actual_data = data["Data"]

    if isinstance(actual_data, list):
        for item in actual_data:
            if isinstance(item, dict):
                # Add metadata
                item["_comcat_id"] = comcat_id
                item["_category"] = cat_name
                item["_start_date"] = start_date
                item["_end_date"] = end_date
                item["_province"] = PROVINCE_NAME
                rows.append(item)
            elif isinstance(item, list):
                # Tabular row — we'll need column headers
                rows.append({
                    "raw_row": item,
                    "_comcat_id": comcat_id,
                    "_category": cat_name,
                })
    elif isinstance(actual_data, dict):
        # Single record or nested structure
        actual_data["_comcat_id"] = comcat_id
        actual_data["_category"] = cat_name
        rows.append(actual_data)

    return rows


def main():
    log.info("=" * 60)
    log.info("PIHPS Production Scraper — South Sumatra")
    log.info(f"Categories: {list(CATEGORIES.values())}")
    log.info(f"Province: {PROVINCE_NAME} (id={PROVINCE_ID})")
    log.info(f"Date range: {START_DATE} to {END_DATE}")
    log.info("=" * 60)

    # Step 1: Establish session
    if not establish_session():
        log.error("Cannot connect to PIHPS. Exiting.")
        return
    time.sleep(DELAY)

    # Step 2: Generate date chunks
    date_chunks = generate_date_chunks(START_DATE, END_DATE, CHUNK_DAYS)
    total_tasks = len(CATEGORIES) * len(date_chunks)
    log.info(f"\nTotal API calls: {len(CATEGORIES)} categories x {len(date_chunks)} date chunks = {total_tasks}")
    log.info(f"Estimated time: {total_tasks * (DELAY + 2) // 60} minutes\n")

    # Step 3: Scrape all combinations
    all_rows = []
    all_raw_responses = []
    task_num = 0
    errors = 0

    for comcat_id, cat_name in CATEGORIES.items():
        log.info(f"\n{'─'*40}")
        log.info(f"Category: {cat_name} ({comcat_id})")
        log.info(f"{'─'*40}")

        for start_date, end_date in date_chunks:
            task_num += 1
            log.info(f"  [{task_num}/{total_tasks}] {start_date} to {end_date}...", )

            data = fetch_grid(comcat_id, start_date, end_date)

            if data is not None:
                # Save raw JSON chunk
                chunk_file = CHUNKS_DIR / f"{comcat_id}_{start_date}_{end_date}.json"
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)

                # Parse into rows
                rows = parse_grid_response(data, comcat_id, cat_name, start_date, end_date)
                all_rows.extend(rows)
                all_raw_responses.append(data)
                log.info(f"    OK — {len(rows)} rows")

                # On first successful response, log the structure for debugging
                if task_num <= 2:
                    log.info(f"    Response type: {type(data).__name__}")
                    if isinstance(data, dict):
                        log.info(f"    Keys: {list(data.keys())}")
                        for key in list(data.keys())[:3]:
                            val_preview = str(data[key])[:150]
                            log.info(f"      {key}: {val_preview}")
                    elif isinstance(data, list) and data:
                        log.info(f"    List of {len(data)} items")
                        log.info(f"    First item: {str(data[0])[:200]}")
            else:
                errors += 1
                log.warning(f"    FAILED")

                # If too many consecutive errors, re-establish session
                if errors >= 3:
                    log.info("  Too many errors, re-establishing session...")
                    establish_session()
                    errors = 0
                    time.sleep(DELAY * 2)

            time.sleep(DELAY)

    # Step 4: Combine and save
    log.info(f"\n{'='*60}")
    log.info(f"SCRAPING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Total rows collected: {len(all_rows)}")
    log.info(f"Total API calls made: {task_num}")
    log.info(f"Errors: {errors}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_path = RAW_DIR / "pihps_prices_all.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"\nData saved to: {output_path}")
        log.info(f"Columns: {list(df.columns)}")
        log.info(f"Shape: {df.shape}")
        log.info(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

        # Also save a summary
        log.info(f"\nColumn value counts:")
        for col in df.columns:
            if df[col].nunique() < 20:
                log.info(f"  {col}: {df[col].nunique()} unique values")
    else:
        log.warning("No data collected! Check the error messages above.")
        log.info("The raw JSON responses are saved in data/raw/pihps_chunks/")
        log.info("Send those files to Claude for debugging.")

    log.info(f"\nRaw JSON chunks saved in: {CHUNKS_DIR}")
    log.info(f"Total chunk files: {len(list(CHUNKS_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
