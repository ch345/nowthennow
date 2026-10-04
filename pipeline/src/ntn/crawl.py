"""Stage 2: polite, incremental crawl of the pages in ``people``.

Politeness: one request at a time per host, bounded concurrency overall, robots.txt and
``noarchive`` honored, a descriptive User-Agent, conditional GET. Raw HTML is stored when the
byte hash is new, including non-content noise such as cache comments. Whether that change is a
new version is decided later, by extract. Logs never contain page content.
"""

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urldefrag, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import polars as pl
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from ntn.config import tuning
from ntn.storage import Storage
from ntn.tables import FETCHES

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
SKIP_ERRORS = {"robots", "noarchive", "denylist"}  # deliberate skips are not failures
_META_NOARCHIVE = re.compile(
    rb"<meta[^>]+name=[\"'](?:robots|googlebot)[\"'][^>]*content=[\"'][^\"']*noarchive", re.I
)


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # sortable; ms matter for tests


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def base_url(url: str) -> str:
    return urldefrag(url)[0]


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


@dataclass(frozen=True)
class Prior:
    """What we know from the last successful fetch of a URL."""

    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None
    raw_object_key: str | None = None


@dataclass(frozen=True)
class CrawlReport:
    run_id: str
    attempted: int
    ok: int
    not_modified: int
    failed: int
    skipped: int
    errors: dict[str, int]  # error code -> row count, skips and failures together

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0

    @property
    def healthy(self) -> bool:
        return self.failure_rate <= tuning().crawl.failure_threshold


def is_denied(url: str, denylist: set[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    base = base_url(url)
    for entry in denylist:
        e = entry.strip().lower()
        if not e:
            continue
        if e in (url.lower(), base.lower()) or host == e or host.endswith("." + e):
            return True
    return False


def priors_from_fetches(fetches: pl.DataFrame | None) -> dict[str, Prior]:
    """Latest usable (200/304 with a content hash) fetch per URL, derived from the data."""
    if fetches is None or fetches.is_empty():
        return {}
    usable = (
        fetches.filter(pl.col("status").is_in([200, 304]) & pl.col("content_hash").is_not_null())
        .sort("fetched_at")
        .group_by("url", maintain_order=True)
        .last()
    )
    # An etag is only valid alongside the body it came with; carry the newest non-null ones.
    return {
        r["url"]: Prior(r["etag"], r["last_modified"], r["content_hash"], r["raw_object_key"])
        for r in usable.iter_rows(named=True)
    }


class Crawler:
    def __init__(
        self,
        client: httpx.AsyncClient,
        bucket: Storage,
        *,
        user_agent: str,
        run_id: str,
        concurrency: int = 20,
        host_delay: float = 1.0,
        retry_wait: float = 1.0,
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.user_agent = user_agent
        self.run_id = run_id
        self.host_delay = host_delay
        self.retry_wait = retry_wait
        self._sem = asyncio.Semaphore(concurrency)
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    # -- public ----------------------------------------------------------------------------

    async def crawl(
        self, urls: list[str], priors: dict[str, Prior], denylist: set[str]
    ) -> list[dict[str, Any]]:
        """Fetch every distinct base URL once; emit one fetch row per listed URL."""
        groups: dict[str, list[str]] = {}
        for u in urls:
            groups.setdefault(base_url(u), []).append(u)

        async def one(base: str, listed: list[str]) -> list[dict[str, Any]]:
            if any(is_denied(u, denylist) for u in listed):
                return [self._row(u, base, error="denylist") for u in listed]
            prior = next((priors[u] for u in listed if u in priors), Prior())
            host = urlsplit(base).netloc.lower()
            lock = self._host_locks.setdefault(host, asyncio.Lock())
            async with lock:
                row = await self._fetch(base, prior)
                if self.host_delay:
                    await asyncio.sleep(self.host_delay)
            return [{**row, "url": u} for u in listed]

        results = await asyncio.gather(*(one(b, us) for b, us in groups.items()))
        return [row for rows in results for row in rows]

    # -- internals -------------------------------------------------------------------------

    def _row(self, url: str, final_url: str | None = None, **kw: Any) -> dict[str, Any]:
        row: dict[str, Any] = {k: None for k in FETCHES}
        row.update(run_id=self.run_id, url=url, fetched_at=now_iso(), final_url=final_url)
        row.update(kw)
        return row

    async def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = None
            try:
                resp = await self.client.get(
                    f"{origin}/robots.txt", headers={"User-Agent": self.user_agent}
                )
                if resp.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(resp.text.splitlines())
                elif resp.status_code in (401, 403):
                    parser = RobotFileParser()
                    parser.parse(["User-agent: *", "Disallow: /"])
            except httpx.HTTPError:
                parser = None  # unreachable robots.txt: allow, the page fetch will tell
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self.user_agent, url)

    async def _get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        async def attempt() -> httpx.Response:
            async with self._sem:
                return await self.client.get(url, headers=headers)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=self.retry_wait),
            retry=retry_if_exception_type(httpx.TransportError)
            | retry_if_result(lambda r: r.status_code in RETRY_STATUSES),
            # After the last attempt, return the last response or re-raise the last exception.
            retry_error_callback=lambda rs: rs.outcome.result() if rs.outcome else None,
        )
        return await retrying(attempt)

    async def _fetch(self, url: str, prior: Prior) -> dict[str, Any]:
        if not await self._allowed(url):
            return self._row(url, url, error="robots")
        headers = {"User-Agent": self.user_agent, "Accept": "text/html,*/*;q=0.8"}
        if prior.etag:
            headers["If-None-Match"] = prior.etag
        if prior.last_modified:
            headers["If-Modified-Since"] = prior.last_modified
        try:
            resp = await self._get(url, headers)
        except httpx.HTTPError as exc:
            return self._row(url, url, error=f"{type(exc).__name__}")
        final = str(resp.url)
        etag = resp.headers.get("etag")
        lastmod = resp.headers.get("last-modified")

        if resp.status_code == 304:
            return self._row(
                url, final, status=304, etag=etag or prior.etag,
                last_modified=lastmod or prior.last_modified,
                content_hash=prior.content_hash, raw_object_key=prior.raw_object_key,
            )  # fmt: skip
        if resp.status_code >= 400 or resp.status_code < 200:
            return self._row(url, final, status=resp.status_code, error=f"http_{resp.status_code}")

        body = resp.content
        ctype = resp.headers.get("content-type", "").lower()
        if ctype and not any(t in ctype for t in ("html", "text/", "xml")):
            return self._row(url, final, status=resp.status_code, error="content_type")
        if len(body) > tuning().crawl.max_bytes:
            return self._row(url, final, status=resp.status_code, error="too_large")
        xrobots = resp.headers.get("x-robots-tag", "").lower()
        if "noarchive" in xrobots or _META_NOARCHIVE.search(body[:200_000]):
            return self._row(url, final, status=resp.status_code, error="noarchive")

        content_hash = hashlib.sha256(body).hexdigest()
        key = f"raw/{url_key(url)[:2]}/{url_key(url)}/{content_hash[:16]}"
        await asyncio.to_thread(self._store_raw, key, body, resp, final)
        return self._row(
            url, final, status=resp.status_code, etag=etag, last_modified=lastmod,
            content_hash=content_hash, bytes=len(body), raw_object_key=key,
        )  # fmt: skip

    def _store_raw(self, key: str, body: bytes, resp: httpx.Response, final: str) -> None:
        if self.bucket.exists(f"{key}.html"):
            return  # content unchanged since a previous run: store only on change
        self.bucket.write_bytes(f"{key}.html", body)
        meta = {
            "final_url": final,
            "status": resp.status_code,
            "headers": {
                k: resp.headers[k]
                for k in ("content-type", "etag", "last-modified", "x-robots-tag", "date")
                if k in resp.headers
            },
        }
        self.bucket.write_json(f"{key}.json", json.dumps(meta))


def summarize(run_id: str, rows: list[dict[str, Any]]) -> CrawlReport:
    skipped = sum(1 for r in rows if r["error"] in SKIP_ERRORS)
    attempted = len(rows) - skipped
    not_modified = sum(1 for r in rows if r["status"] == 304)
    failed = sum(1 for r in rows if r["error"] and r["error"] not in SKIP_ERRORS)
    errors = dict(Counter(r["error"] for r in rows if r["error"]))
    return CrawlReport(
        run_id, attempted, attempted - failed - not_modified, not_modified, failed, skipped, errors
    )


async def run_crawl(
    dataset: Storage,
    bucket: Storage,
    *,
    user_agent: str,
    limit: int | None = None,
    concurrency: int = 20,
    host_delay: float = 1.0,
    retry_wait: float = 1.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CrawlReport:
    people = dataset.read_table("people")
    if people is None:
        raise RuntimeError("no people table: run `ntn directory` first")
    urls = sorted(people.filter(pl.col("valid_to").is_null())["url"].unique().to_list())
    if limit:
        urls = urls[:limit]
    denylist = {
        ln.strip() for ln in (dataset.read_text("denylist.txt") or "").splitlines() if ln.strip()
    }
    priors = priors_from_fetches(dataset.read_tables("fetches"))
    run_id = new_run_id()
    started = now_iso()

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=20, transport=transport, max_redirects=5
    ) as client:
        crawler = Crawler(
            client, bucket, user_agent=user_agent, run_id=run_id, concurrency=concurrency,
            host_delay=host_delay, retry_wait=retry_wait,
        )  # fmt: skip
        rows = await crawler.crawl(urls, priors, denylist)

    dataset.write_table(f"fetches/{run_id}", pl.DataFrame(rows, schema=FETCHES))
    report = summarize(run_id, rows)
    dataset.write_json(
        f"runs/{run_id}-crawl.json",
        json.dumps({"stage": "crawl", "started_at": started, "healthy": report.healthy,
                    **asdict(report)}),
    )  # fmt: skip
    log.info("crawl %s: %s healthy=%s", run_id, asdict(report), report.healthy)
    return report
