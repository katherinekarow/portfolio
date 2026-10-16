"""The browser calculator gives the same answers as the Python package, for every formula."""
import functools
import http.server
import threading
from pathlib import Path

import pytest

from formulakit import REGISTRY

DOCS = Path(__file__).resolve().parents[3] / "docs"
pw = pytest.importorskip("playwright.sync_api")


def cases(f):
    base = f.defaults()
    out = [base]
    for k in (0.9, 1.1):
        c = dict(base)
        for p in f.params:
            if p.kind == "num" and p.name not in ("q",):
                c[p.name] = round(float(p.default) * k, 6)
            elif p.kind == "list":
                c[p.name] = ",".join(str(round(float(x) * k, 6)) for x in str(p.default).split(","))
        out.append(c)
    return out


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if a is None or b is None or isinstance(a, bool):
        return a == b
    return abs(a - b) <= 1e-9 * max(1, abs(a), abs(b))


@pytest.fixture(scope="module")
def page():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with pw.sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(m.text))  # web fonts may be offline in CI
        pg.goto(f"http://127.0.0.1:{srv.server_port}/formulas/")
        pg.wait_for_function("window.__ready === true")
        pg.errors = errors
        yield pg
        br.close()
    srv.shutdown()


def test_page_loads_every_formula(page):
    assert not page.errors, page.errors
    assert page.locator(".fcard").count() == len(REGISTRY)
    assert "No solution" not in page.locator("#f-npv .result").inner_text()


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_js_matches_python(page, key):
    f = REGISTRY[key]
    for raw in cases(f):
        py = f.run(raw)
        js = page.evaluate("([k, raw]) => FK.run(SPECS.find(s => s.key === k), raw)", [key, {k: str(v) for k, v in raw.items()}])
        assert close(py, js), (key, raw, py, js)
