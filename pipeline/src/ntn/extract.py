"""Stage 3: raw HTML -> normalized markdown, versioned.

A new version is created only when the normalized text changes meaningfully. Unchanged fetches
just advance ``last_seen`` on the latest version. Logs never contain page content.
"""

import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urldefrag

import dateparser
import polars as pl
import trafilatura
from lxml import html as lxml_html

from ntn.config import tuning
from ntn.crawl import new_run_id
from ntn.storage import Storage
from ntn.tables import VERSIONS, empty

log = logging.getLogger(__name__)


_VOLATILE_LINE = re.compile(
    r"^\W*(?:this page was\s+)?(?:last\s+)?(?:updated|modified|edited)\b.{0,60}$", re.I
)
_COUNTER = re.compile(r"\b\d[\d,]*\s+(?:visitors?|views?|hits|visits)\b", re.I)
_STATED = re.compile(
    r"(?:last\s+)?(?:updated|modified|edited)\s*(?:on|:|-)?\s*([^\n|]{4,40})", re.I
)


def extract_markdown(html: bytes, url: str) -> str:
    """Main text as markdown. A ``#fragment`` in ``url`` selects the element with that id."""
    source: str | bytes = html
    fragment = urldefrag(url)[1]
    if fragment:
        try:
            tree = lxml_html.fromstring(html)
            nodes = tree.xpath("//*[@id=$i]", i=fragment)
            if nodes:
                inner = lxml_html.tostring(nodes[0], encoding="unicode", with_tail=False)
                source = f"<html><body>{inner}</body></html>"
        except (ValueError, lxml_html.etree.ParserError):
            pass
    text = trafilatura.extract(
        source, output_format="markdown", include_comments=False, include_tables=True, url=None
    )
    return text or ""


def parse_stated_date(text: str) -> str | None:
    """The page's own "last updated" date, as ISO ``YYYY-MM-DD``, if it states one."""
    for m in _STATED.finditer(text):
        parsed = dateparser.parse(
            m.group(1).strip(" .,:;"), settings={"PREFER_DAY_OF_MONTH": "first"}
        )
        if parsed:
            return parsed.date().isoformat()
    return None


def normalize(text: str) -> str:
    """Strip volatile lines (last-updated stamps, counters) and collapse whitespace."""
    out: list[str] = []
    for line in text.splitlines():
        line = _COUNTER.sub("", line).rstrip()
        if _VOLATILE_LINE.match(line.strip()):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def is_meaningful_change(old: str, new: str) -> bool:
    if old == new:
        return False
    a, b = old.splitlines(), new.splitlines()
    return (
        difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        < tuning().extract.meaningful_ratio
    )


@dataclass(frozen=True)
class ExtractReport:
    examined: int
    new_versions: int
    unchanged: int
    needs_js: int
    errors: int


def _version_row(
    url: str, person_id: str | None, fetched_at: str, text: str, stated: str | None,
    content_hash: str,
) -> dict[str, Any]:  # fmt: skip
    th = text_hash(text)
    return {
        # first_seen is part of the id so a page reverting to old text is a new version
        "version_id": hashlib.sha1(f"{url}\0{th}\0{fetched_at}".encode()).hexdigest(),
        "person_id": person_id,
        "url": url,
        "first_seen": fetched_at,
        "last_seen": fetched_at,
        "source": "crawl",
        "archive_url": None,
        "stated_date": stated,
        "text_md": text,
        "language": None,
        "richness": None,
        "content_hash": content_hash,
        "text_hash": th,
        "needs_js": len(text) < tuning().extract.min_text_chars,
    }


def run_extract(dataset: Storage, bucket: Storage) -> ExtractReport:
    fetches = dataset.read_tables("fetches")
    if fetches is None:
        raise RuntimeError("no fetches: run `ntn crawl` first")
    people = dataset.read_table("people")
    pid_by_url: dict[str, str] = {}
    if people is not None:
        for r in people.sort("valid_from").iter_rows(named=True):
            pid_by_url[r["url"]] = r["person_id"]

    existing = dataset.read_table("versions")
    versions = existing if existing is not None else empty(VERSIONS)
    latest: dict[str, dict[str, Any]] = {}
    for r in versions.sort("last_seen").iter_rows(named=True):
        latest[r["url"]] = r  # latest version per url

    usable = fetches.filter(
        pl.col("content_hash").is_not_null() & pl.col("status").is_in([200, 304])
    ).sort(["url", "fetched_at"])

    new_rows: list[dict[str, Any]] = []
    updates: dict[str, dict[str, Any]] = {}  # version_id -> changed fields on existing rows
    examined = unchanged = needs_js = errors = 0

    for f in usable.iter_rows(named=True):
        url, at = f["url"], f["fetched_at"]
        cur = latest.get(url)
        if cur is not None and at <= cur["last_seen"]:
            continue  # already accounted for
        examined += 1
        if cur is not None and (f["status"] == 304 or f["content_hash"] == cur["content_hash"]):
            cur["last_seen"] = at
            updates[cur["version_id"]] = {**updates.get(cur["version_id"], {}), "last_seen": at}
            unchanged += 1
            continue
        try:
            raw = bucket.read_bytes(f"{f['raw_object_key']}.html")
            markdown = extract_markdown(raw, url)
            text, stated = normalize(markdown), parse_stated_date(markdown)
        except Exception as exc:  # one bad page must not stop the run
            log.warning("extract failed for %s: %s", url, type(exc).__name__)
            errors += 1
            continue
        if cur is not None and not is_meaningful_change(cur["text_md"], text):
            cur["last_seen"], cur["content_hash"] = at, f["content_hash"]
            updates[cur["version_id"]] = {
                **updates.get(cur["version_id"], {}),
                "last_seen": at,
                "content_hash": f["content_hash"],
            }
            unchanged += 1
            continue
        row = _version_row(url, pid_by_url.get(url), at, text, stated, f["content_hash"])
        if row["needs_js"]:
            needs_js += 1
        new_rows.append(row)
        latest[url] = row

    if updates:
        rows = [{**r, **updates.get(r["version_id"], {})} for r in versions.iter_rows(named=True)]
        versions = pl.DataFrame(rows, schema=VERSIONS)
    if new_rows:
        versions = pl.concat([versions, pl.DataFrame(new_rows, schema=VERSIONS)])
    dataset.write_table("versions", versions)

    report = ExtractReport(examined, len(new_rows), unchanged, needs_js, errors)
    run_id = new_run_id()
    dataset.write_json(
        f"runs/{run_id}-extract.json", json.dumps({"stage": "extract", **report.__dict__})
    )
    log.info("extract: %s", report)
    return report
