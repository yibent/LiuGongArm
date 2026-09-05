"""Optional headless UI smoke; run against an already-started local WebUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7863")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/vision_webui"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1080})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url)
            try:
                page.wait_for_load_state("networkidle", timeout=2000)
            except PlaywrightTimeoutError:
                # This page polls status every 400 ms, so a 500-ms network idle
                # window is not guaranteed. Check the rendered config explicitly.
                page.wait_for_function("document.querySelector('#prompt').value === 'bus'")
            print(json.dumps(page.locator("button, input, select, textarea").evaluate_all(
                "els => els.map(e => ({tag:e.tagName, id:e.id, text:e.textContent.trim(), value:e.value}))"
            ), ensure_ascii=False), flush=True)
            page.screenshot(path=str(args.output / "initial.png"), full_page=True)
            if args.inspect_only:
                return
            # These IDs were confirmed from the rendered --inspect-only result.
            assert page.locator("#prompt").input_value() == "bus"
            assert page.locator("#backend").input_value() == "cv"
            page.wait_for_function("document.querySelector('#loadDot').classList.contains('ok')", timeout=90000)
            page.locator("#sampleBtn").click()
            page.wait_for_function("document.querySelector('#fileName').textContent.includes('bus.jpg')")
            page.locator("#startBtn").click()
            page.wait_for_function("document.querySelector('#rows').textContent.includes('bus')", timeout=90000)
            status = page.request.get(args.url + "/api/status").json()
            assert status["florence"] and not status["yoloe"], status
            assert status["config"]["fast_backend"] == "cv", status
            assert status["detections"] and not status["load_error"], status
            assert not errors, errors
            # Do not capture the earlier 854x480 placeholder while the MJPEG
            # response is still presenting its first real 810x1080 sample frame.
            page.wait_for_function("document.querySelector('#view').naturalHeight === 1080", timeout=20000)
            page.screenshot(path=str(args.output / "result.png"), full_page=True)
            (args.output / "report.json").write_text(json.dumps({"passed": True, "page_errors": errors, "status": status}, ensure_ascii=False, indent=2), encoding="utf-8")
            print("WebUI CV + local Florence inference passed", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
