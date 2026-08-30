"""Take the deck's three station screenshots.

    uv run python -m aoi_agent station --port 8111 &
    uv run --with playwright python scripts/deck_screenshots.py

Credentials come from the operator file; the demo account is the senior one.
"""
from playwright.sync_api import sync_playwright
import pathlib, socket, time
OUT = pathlib.Path("/Users/lin1020/Projects/aoi-agent/docs/deck/img")
for _ in range(60):
    try: socket.create_connection(("127.0.0.1", 8111), timeout=1).close(); break
    except OSError: time.sleep(1)
BASE = "http://127.0.0.1:8111"   # start one with: uv run python -m aoi_agent station --port 8111
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 2000}, color_scheme="dark", device_scale_factor=3)
    pg = ctx.new_page()
    pg.goto(BASE + "/login"); pg.fill("input[name=name]", "mike"); pg.fill("input[name=secret]", "0000")
    pg.click("button[type=submit]"); pg.wait_for_load_state("networkidle")
    pg.goto(BASE + "/locale/zh-TW?next=/"); pg.wait_for_load_state("networkidle")

    def band(name, path, js_top, js_bottom):
        pg.goto(BASE + path); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(600)
        top = pg.evaluate(js_top); bottom = pg.evaluate(js_bottom)
        h = max(120, bottom - top)
        # full_page, or the clip is silently truncated at the viewport's bottom
        # edge and the shot ends mid-content with no error.
        pg.screenshot(path=str(OUT / name), full_page=True,
                      clip={"x": 0, "y": top, "width": 1200, "height": h})
        print(f"{name}: {1200}x{round(h)}  aspect {1200 / h:.2f}")

    # The queue: from the top of the page to the end of the second row.
    band("shot_queue.png", "/", "0",
         "document.querySelectorAll('table.queue tbody tr')[1].getBoundingClientRect().bottom + window.scrollY")
    # The region and the chart are one element each: Playwright scrolls to
    # them and cuts at their own edges, so nothing is clipped mid-content.
    def element(name, path, selector):
        pg.goto(BASE + path); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(600)
        pg.locator(selector).first.screenshot(path=str(OUT / name))
        print(name, "element", selector)

    element("shot_region.png", "/c/20085299/2", "figure.triptych")
    # The seven answers and the eighth button. The triptych alone is the
    # evidence; this is the thing the operator actually presses, and the
    # deferral button is one of the deck's own stories.
    band("shot_verdict.png", "/c/20085299/2",
         "document.querySelector('form.verdict').getBoundingClientRect().top + window.scrollY - 6",
         "document.querySelector('form.defer').getBoundingClientRect().bottom + window.scrollY + 12")
    element("shot_ask.png", "/ask/36", "figure.chart")
    b.close()
