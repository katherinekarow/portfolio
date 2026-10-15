"""Every dashboard page loads without errors, and the two in-browser calculators agree with Python."""
import functools
import http.server
import json
import threading
from pathlib import Path

import pytest

from dashkit import builders as b

DOCS = Path(__file__).resolve().parents[3] / "docs"
PAGES = ["", "fill-pace/", "price-performance/", "outreach-effectiveness/", "market-review/",
         "same-week-openings/", "group-quote/", "comparables/"]

pw = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def server():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/dashboards/"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        br = p.chromium.launch()
        yield br
        br.close()


@pytest.mark.parametrize("page_path", PAGES)
def test_page_loads_cleanly(server, browser, page_path):
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(server + page_path)
    if page_path:
        page.wait_for_function("window.__ready === true || document.querySelector('#kpis .kpi')", timeout=15000)
        assert page.locator("svg").count() > 0
    else:
        assert page.locator("a.card").count() == len(PAGES) - 1
    assert not errors, errors
    page.close()


def test_quote_parity(server, browser):
    page = browser.new_page()
    page.goto(server + "group-quote/")
    page.wait_for_function("window.__ready === true")
    params = b.group_quote_inputs()
    for s in params["sites"]:
        for w in s["weeks"][::3]:
            open_slots = w["capacity"] - w["booked"]
            for n in sorted({1, max(1, open_slots // 2), open_slots, open_slots + 1}):
                js = page.evaluate("([s, w, n]) => quote(window.P, s, w, n)", [s["site"], w["week"], n])
                assert js == b.quote(params, s["site"], w["week"], n)
    page.close()


def test_nearest_parity(server, browser):
    page = browser.new_page()
    page.goto(server + "comparables/")
    page.wait_for_function("window.__ready === true")
    data = json.loads((DOCS / "dashboards" / "comparables" / "data.json").read_text())
    for reach in ("regional", "local"):
        for season in ("flu season", "off-season"):
            for cap in (1200, 2100, 3000):
                py = [s["id"] for s in b.nearest(data, reach, season, "one-day", cap, k=8)]
                js = page.evaluate("([r, s, c]) => nearest(window.DATA, r, s, 'one-day', c, 8).map(x => x.id)", [reach, season, cap])
                assert js == py
    page.close()
