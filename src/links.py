"""Link verification: does a job posting's URL still open a real page?

Encodes a hard rule of this project — never surface a job whose link is dead. A
checker hits each posting's URL and classifies it; the store records the result and
the cockpit drops known-dead postings out of the "apply today" queue.

Classification is deliberately conservative:
  alive   - 2xx/3xx (the page or its redirect resolves)
  dead    - 404 / 410 (gone for good)
  unknown - 401/403/429/5xx/timeout/DNS — ambiguous (bot-walls, rate limits); flag, don't kill

The network call is injectable (`fetcher`) so tests run offline and deterministically.
The default fetcher is SSRF-guarded: https-only, and the host must not resolve to a
private / loopback / link-local address.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

DEAD_CODES = {404, 410}


def classify(status: int | None) -> str:
    if status is None:
        return "unknown"
    if 200 <= status < 400:
        return "alive"
    if status in DEAD_CODES:
        return "dead"
    return "unknown"


class LinkError(Exception):
    """Raised by a fetcher when a request can't complete (DNS, timeout, blocked host)."""


def _default_fetcher(url: str, *, timeout: float = 12.0) -> int:
    """SSRF-guarded HEAD (falling back to GET) returning the HTTP status code."""
    import httpx

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LinkError("non-http(s) or hostless URL")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise LinkError(f"DNS failure: {exc}") from exc
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise LinkError("URL resolves to a disallowed (internal) address")
    headers = {"User-Agent": "jobproof-linkcheck/1.0 (+https://github.com/bitcoinbox/jobproof)"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as c:
            resp = c.head(url)
            # Some hosts don't implement HEAD well (405/501) — confirm with a light GET.
            if resp.status_code in (403, 405, 501):
                resp = c.get(url)
            return resp.status_code
    except httpx.HTTPError as exc:
        raise LinkError(str(exc)) from exc


def check_url(url: str | None, *, fetcher=None) -> dict:
    """Check one URL. Returns {url, status, link_status, note}. Never raises."""
    if not url:
        return {"url": url, "status": None, "link_status": "unknown", "note": "no url"}
    fetcher = fetcher or _default_fetcher
    try:
        status = fetcher(url)
    except LinkError as exc:
        return {"url": url, "status": None, "link_status": "unknown", "note": str(exc)}
    except Exception as exc:  # defensive: a checker should never crash a batch
        return {"url": url, "status": None, "link_status": "unknown", "note": f"{type(exc).__name__}: {exc}"}
    return {"url": url, "status": status, "link_status": classify(status), "note": ""}


def check_jobs(conn, *, fetcher=None, only_unchecked: bool = False, limit: int | None = None) -> dict:
    """Check stored jobs that have a URL; persist each result. Returns a summary dict."""
    from . import store

    rows = [r for r in store.dashboard_rows(conn) if r["url"]]
    if only_unchecked:
        rows = [r for r in rows if r["link_status"] == "unchecked"]
    if limit is not None:
        rows = rows[:limit]
    counts = {"alive": 0, "dead": 0, "unknown": 0}
    dead: list[dict] = []
    for r in rows:
        result = check_url(r["url"], fetcher=fetcher)
        store.set_link_status(conn, r["id"], result["link_status"])
        counts[result["link_status"]] = counts.get(result["link_status"], 0) + 1
        if result["link_status"] == "dead":
            dead.append({"job_id": r["id"], "company": r["company"], "title": r["title"], "url": r["url"]})
    return {"checked": len(rows), **counts, "dead_jobs": dead}
