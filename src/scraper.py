"""Playwright scraper for the carrier portal (portal must be running on :8001).
CLI:  python -m src.scraper OCLU1023757 BWSU1007919 XXXX0000000 [--headed]"""
import asyncio
import json
import sys
from datetime import datetime, timezone

from playwright.async_api import Browser, TimeoutError as PWTimeout, async_playwright

from src import config

STATE_FILE = config.PORTAL_STATE_FILE  # saved cookies = "logged-in session"


async def _login(browser: Browser) -> None:
    """Log in once and save the session (cookies) to STATE_FILE."""
    context = await browser.new_context()   # fresh "incognito profile"
    page = await context.new_page()         # one tab
    await page.goto(f"{config.PORTAL_URL}/login")
    await page.fill("#username", config.PORTAL_USER)        # auto-waits until the input exists
    await page.fill("#password", config.PORTAL_PASSWORD)
    await page.click("#login-btn")
    await page.wait_for_url("**/track", timeout=10_000)      # fails fast if login was rejected
    await context.storage_state(path=str(STATE_FILE))        # save cookies for reuse
    await context.close()


async def _track_one(browser: Browser, container: str, sem: asyncio.Semaphore) -> dict:
    """Scrape one container. Errors are returned as data, never raised."""
    async with sem:  # at most N pages at once = polite to the site
        context = await browser.new_context(storage_state=str(STATE_FILE))  # already logged in
        page = await context.new_page()
        try:
            await page.goto(f"{config.PORTAL_URL}/track")
            if "/login" in page.url:  # portal threw us out -> cookie no longer valid
                return {"container_no": container, "error": "session_expired"}
            await page.fill("#container-input", container)
            await page.click("#search-btn")

            result, not_found = page.locator("#result"), page.locator("#not-found")
            await result.or_(not_found).wait_for(timeout=10_000)  # whichever appears first
            if await not_found.is_visible():
                return {"container_no": container, "error": "not_found"}

            events = []
            for row in await page.locator("tr.event").all():
                time, code, name, location = await row.locator("td").all_inner_texts()
                events.append({"time": time, "code": code, "event": name, "location": location})
            return {
                "container_no": container,
                "live_eta": await page.locator("#eta").inner_text(),
                "latest_status": events[-1]["code"] if events else None,
                "events": events,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        except PWTimeout:
            return {"container_no": container, "error": "timeout"}
        finally:
            await context.close()


async def track_containers(containers: list[str], headless: bool = True, concurrency: int = 3) -> list[dict]:
    containers = [c.strip().upper() for c in containers]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        try:
            if not STATE_FILE.exists():
                await _login(browser)
            sem = asyncio.Semaphore(concurrency)
            results = await asyncio.gather(*[_track_one(browser, c, sem) for c in containers])

            # Session expired? Log in again ONCE and retry only those containers.
            expired = [i for i, r in enumerate(results) if r.get("error") == "session_expired"]
            if expired:
                await _login(browser)
                retried = await asyncio.gather(*[_track_one(browser, containers[i], sem) for i in expired])
                for i, r in zip(expired, retried):
                    results[i] = r
            return list(results)
        finally:
            await browser.close()


def track_containers_sync(containers: list[str], headless: bool = True) -> list[dict]:
    """For normal (non-async) code, e.g. agent tools."""
    return asyncio.run(track_containers(containers, headless=headless))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = track_containers_sync(args or ["OCLU1023757"], headless="--headed" not in sys.argv)
    print(json.dumps(out, indent=2))
