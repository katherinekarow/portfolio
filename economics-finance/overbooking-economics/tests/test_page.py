"""The web calculator loads cleanly and its JavaScript returns the same answer as the Python reference."""
import functools
import http.server
import itertools
import json
import threading
from pathlib import Path

import pytest

from overbook import optimal_limit

DOCS = Path(__file__).resolve().parents[3] / "docs"
pw = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def server():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/overbooking/"
    srv.shutdown()


@pytest.fixture(scope="module")
def page(server):
    with pw.sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
        pg.goto(server)
        pg.wait_for_function("window.__ready === true", timeout=15000)
        pg.errors = errors
        yield pg
        br.close()


def test_page_loads_without_console_errors(page):
    assert page.locator("#kpis .kpi").count() == 6
    assert page.locator("svg.chart path").count() >= 3
    assert page.locator("#presets button").count() >= 4
    page.locator("#presets button").nth(3).click()          # a preset re-renders without errors
    page.fill("#rho", "0.08")
    page.set_viewport_size({"width": 390, "height": 800})   # phone width: no horizontal scroll
    page.wait_for_timeout(400)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    assert not page.errors, page.errors


GRID = list(itertools.product([5, 20, 60], [0.62, 0.85, 0.95], [(150, 60), (150, 300), (90, 900)], [0.0, 0.05], [0.0, 18.0]))


def test_js_optimum_equals_python(page):
    cases = [[c, p, r, b, rho, lam] for c, p, (r, b), rho, lam in GRID]
    js = page.evaluate("cs => cs.map(([C,p,r,b,rho,lam]) => { const o = overbook.optimalLimit(C,p,r,b,rho,lam); return [o.limit, o.profit, o.bumps, o.empty]; })", cases)
    for (c, p, r, b, rho, lam), (L, prof, bumps, empty) in zip(cases, js):
        py = optimal_limit(c, p, r, b, rho, lam or None)
        assert L == py["limit"], (c, p, r, b, rho, lam)
        assert prof == pytest.approx(py["profit"], rel=1e-10, abs=1e-8)
        assert bumps == pytest.approx(py["bumps"], rel=1e-9, abs=1e-10)
        assert empty == pytest.approx(py["empty"], rel=1e-9, abs=1e-10)


def test_page_shows_python_answer_for_each_preset(page):
    data = json.loads((DOCS / "overbooking" / "data.json").read_text())
    page.set_viewport_size({"width": 1200, "height": 900})
    for i, pr in enumerate(data["presets"]):
        page.locator("#presets button").nth(i).click()
        last = page.evaluate("window.__last")
        assert last["limit"] == pr["python"]["limit"], pr["name"]
        assert last["profit"] == pytest.approx(pr["python"]["profit"], abs=1e-5)
        assert page.locator("#kpis .kpi .v").first.inner_text() == str(pr["python"]["limit"])
