"""
01a_discover_pihps.py — Discovery script: load PIHPS and inspect the page
==========================================================================
Run this FIRST. It opens the PIHPS page in a visible browser window so
you can see exactly what it looks like, takes a screenshot, and dumps
all form elements it finds.

USAGE:
    python scripts/01a_discover_pihps.py
"""

import time
import json
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalKomoditas"


def main():
    log.info("=" * 60)
    log.info("PIHPS Discovery — Loading page in VISIBLE browser")
    log.info("=" * 60)

    with sync_playwright() as pw:
        # Launch VISIBLE browser so you can see what's happening
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Track XHR data requests (not CSS/JS)
        data_requests = []
        def on_response(response):
            url = response.url
            ct = response.headers.get("content-type", "")
            # Only capture JSON or HTML data responses, skip CSS/JS/images/fonts
            if any(skip in ct for skip in ["css", "javascript", "font", "image", "svg"]):
                return
            if any(skip in url for skip in [".css", ".js", ".svg", ".png", ".jpg", ".woff", ".ttf"]):
                return
            try:
                body = response.text()
                if len(body) > 200:
                    data_requests.append({
                        "url": url,
                        "method": response.request.method,
                        "content_type": ct,
                        "body_length": len(body),
                    })
                    log.info(f"  DATA XHR: {response.request.method} {url[:80]}... ({len(body)} chars)")
            except Exception:
                pass

        page.on("response", on_response)

        # Load with generous timeout and relaxed wait condition
        log.info("Loading PIHPS page (this takes 30-60 seconds)...")
        try:
            page.goto(BASE_URL, timeout=90000, wait_until="domcontentloaded")
        except Exception as e:
            log.warning(f"Initial load warning (may still work): {e}")

        # Give JS time to render the page
        log.info("Waiting for page JavaScript to finish rendering...")
        time.sleep(10)

        # Take screenshot
        screenshot_path = RAW_DIR / "pihps_page_screenshot.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        log.info(f"Screenshot saved: {screenshot_path}")

        # Dump all form elements
        log.info("\n--- FORM ELEMENTS FOUND ---")

        # Select dropdowns
        selects = page.query_selector_all("select")
        log.info(f"\n<select> elements: {len(selects)}")
        for i, sel in enumerate(selects):
            sel_id = sel.get_attribute("id") or "no-id"
            sel_name = sel.get_attribute("name") or "no-name"
            options = sel.query_selector_all("option")
            opt_texts = [o.text_content().strip() for o in options[:8]]
            log.info(f"  [{i}] id='{sel_id}' name='{sel_name}' | options: {opt_texts}")

        # Input fields
        inputs = page.query_selector_all("input")
        log.info(f"\n<input> elements: {len(inputs)}")
        for i, inp in enumerate(inputs):
            inp_id = inp.get_attribute("id") or "no-id"
            inp_name = inp.get_attribute("name") or "no-name"
            inp_type = inp.get_attribute("type") or "text"
            inp_val = inp.get_attribute("value") or ""
            inp_placeholder = inp.get_attribute("placeholder") or ""
            log.info(f"  [{i}] id='{inp_id}' name='{inp_name}' type='{inp_type}' value='{inp_val}' placeholder='{inp_placeholder}'")

        # Buttons
        buttons = page.query_selector_all("button, input[type='submit'], a.btn")
        log.info(f"\nButtons/submit elements: {len(buttons)}")
        for i, btn in enumerate(buttons):
            btn_text = btn.text_content().strip()[:50] if btn.text_content() else ""
            btn_id = btn.get_attribute("id") or "no-id"
            btn_class = btn.get_attribute("class") or ""
            log.info(f"  [{i}] id='{btn_id}' text='{btn_text}' class='{btn_class[:60]}'")

        # DevExtreme widgets (PIHPS uses this framework)
        dx_widgets = page.query_selector_all("[class*='dx-'], [id*='dx']")
        log.info(f"\nDevExtreme widgets: {len(dx_widgets)}")
        for i, w in enumerate(dx_widgets[:15]):  # first 15 only
            w_id = w.get_attribute("id") or "no-id"
            w_class = w.get_attribute("class") or ""
            w_role = w.get_attribute("role") or ""
            log.info(f"  [{i}] id='{w_id}' role='{w_role}' class='{w_class[:80]}'")

        # Save all findings
        findings = {
            "selects": len(selects),
            "inputs": len(inputs),
            "buttons": len(buttons),
            "dx_widgets": len(dx_widgets),
            "data_requests": data_requests,
            "page_title": page.title(),
            "page_url": page.url,
        }
        findings_path = RAW_DIR / "pihps_discovery.json"
        with open(findings_path, "w") as f:
            json.dump(findings, f, indent=2, default=str)
        log.info(f"\nFindings saved: {findings_path}")

        # Now let's try to manually interact and capture the data endpoint
        log.info("\n--- ATTEMPTING FORM INTERACTION ---")
        log.info("The browser window is open. You can also interact with it manually!")
        log.info("I'll try to fill in the form programmatically...")

        try:
            # Try selecting commodity via DevExtreme selectbox
            # DevExtreme uses dx-selectbox or dx-lookup widgets, not plain <select>
            commodity_selectors = [
                "#komoditas", "#Komoditas", "[name='komoditas']",
                ".dx-selectbox:first-of-type", 
                "div.dx-selectbox",
                "[aria-label*='Komoditas']",
            ]
            
            for selector in commodity_selectors:
                el = page.query_selector(selector)
                if el:
                    log.info(f"  Found commodity element with selector: {selector}")
                    # Try clicking to open dropdown
                    el.click()
                    time.sleep(2)
                    # Take screenshot of opened dropdown
                    page.screenshot(path=str(RAW_DIR / "pihps_dropdown_open.png"), full_page=True)
                    log.info(f"  Dropdown screenshot saved")
                    break
            else:
                log.info("  Could not find commodity selector automatically")
                log.info("  Check the screenshot and the discovery.json for clues")
        except Exception as e:
            log.warning(f"  Form interaction failed: {e}")

        log.info("\n" + "=" * 60)
        log.info("DISCOVERY COMPLETE")
        log.info("=" * 60)
        log.info(f"1. Check screenshot: {screenshot_path}")
        log.info(f"2. Check findings:   {findings_path}")
        log.info(f"3. The browser is still open — you can interact with it manually")
        log.info(f"   Try filling in the form yourself and watch the Network tab")
        log.info(f"\nPress Enter to close the browser...")
        
        input()
        browser.close()


if __name__ == "__main__":
    main()
