"""Fake carrier tracking portal (our Playwright target). Login -> track container -> events table.
Run: uvicorn mock_portal.app:app --port 8001"""
import csv
import hashlib
import html
import secrets
from datetime import datetime, timedelta

from fastapi import Cookie, FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from src import config

app = FastAPI(title="Mock Carrier Portal")
SESSIONS: set[str] = set()  # valid session tokens (in memory = lost on restart, like a real expiry)


def _parse_date(text: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y"):  # the 3 formats found in Step 4
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"bad date: {text}")


# Load containers once: {CONTAINER_NO: row}
with open(config.RAW_DIR / "erp_shipments.csv", newline="", encoding="utf-8") as f:
    CONTAINERS = {r["container_no"].strip().upper(): r for r in csv.DictReader(f)}


def _seed(container: str) -> int:
    # hashlib (not hash()) -> same number every run = deterministic portal data
    return int(hashlib.md5(container.encode()).hexdigest(), 16)


def _tracking(container: str) -> tuple[datetime, list[tuple]]:
    row, seed = CONTAINERS[container], _seed(container)
    etd = _parse_date(row["etd"])
    live_eta = _parse_date(row["current_eta"]) + timedelta(days=seed % 4)  # +0..3 days
    events = [
        (etd - timedelta(days=2) + timedelta(hours=9), "GIN", "Gate in full", row["pol"]),
        (etd + timedelta(hours=6), "LOD", "Loaded on vessel", row["pol"]),
        (etd + timedelta(hours=18), "DEP", "Vessel departed", row["pol"]),
    ]
    if seed % 2:  # some containers get a transshipment event
        events.append((etd + timedelta(days=5), "TSP", "Transshipment", "Port Klang"))
    return live_eta, events


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(f"<html><head><title>Carrier Portal</title></head><body>{body}</body></html>")


@app.get("/")
def home():
    return RedirectResponse("/login")


@app.get("/login")
def login_page(error: str = ""):
    msg = f'<p id="error" style="color:red">{error}</p>' if error else ""
    return _page(f"""<h1>Carrier Portal Login</h1>{msg}
        <form method="post" action="/login">
          <input id="username" name="username" placeholder="Username">
          <input id="password" name="password" type="password" placeholder="Password">
          <button id="login-btn" type="submit">Log in</button>
        </form>""")


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    ok = secrets.compare_digest(username, config.PORTAL_USER) and secrets.compare_digest(password, config.PORTAL_PASSWORD)
    if not ok:
        return RedirectResponse("/login?error=Invalid+credentials", status_code=303)
    token = secrets.token_hex(16)
    SESSIONS.add(token)
    resp = RedirectResponse("/track", status_code=303)  # 303 = "go GET this page" after a POST
    resp.set_cookie("session", token, httponly=True)    # httponly: JavaScript can't read the cookie
    return resp


@app.get("/track")
def track(container: str = "", session: str | None = Cookie(default=None)):
    if session not in SESSIONS:
        return RedirectResponse("/login", status_code=303)
    container = html.escape(container.strip().upper())  # never echo user input raw into HTML (XSS)
    form = f"""<h1>Track a Container</h1>
        <form method="get" action="/track">
          <input id="container-input" name="container" value="{container}" placeholder="e.g. OCLU1023757">
          <button id="search-btn" type="submit">Search</button>
        </form>"""
    if not container:
        return _page(form)
    if container not in CONTAINERS:
        return _page(form + f'<p id="not-found">No container found: {container}</p>')

    live_eta, events = _tracking(container)
    rows = "".join(
        f'<tr class="event"><td>{t:%Y-%m-%d %H:%M}</td><td>{code}</td><td>{name}</td><td>{loc}</td></tr>'
        for t, code, name, loc in events
    )
    return _page(form + f"""<div id="result">
        <h2>{container}</h2>
        <p>Estimated arrival: <span id="eta">{live_eta:%Y-%m-%d}</span></p>
        <table id="events"><tr><th>Time</th><th>Code</th><th>Event</th><th>Location</th></tr>{rows}</table>
        </div>""")
