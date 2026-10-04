"""Stage 1: fetch nownownow.txt, snapshot it, diff it, and update ``people``."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import date

import httpx
import polars as pl

from ntn.storage import Storage
from ntn.tables import DIRECTORY_SNAPSHOT, PEOPLE, empty

log = logging.getLogger(__name__)

FIELDS = ["name", "city", "state", "country", "created", "updated", "checked", "url"]
COMPARED = ["name", "city", "state", "country"]  # "checked" is not trusted, dates are noisy


@dataclass(frozen=True)
class DirectoryRow:
    name: str
    city: str
    state: str
    country: str
    created: str
    updated: str
    checked: str
    url: str


@dataclass(frozen=True)
class DirectoryDiff:
    new: list[str]
    changed: list[str]
    removed: list[str]


def parse_directory(text: str) -> list[DirectoryRow]:
    """Parse the tab-separated directory. Malformed rows and duplicate URLs are skipped."""
    rows: dict[str, DirectoryRow] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < len(FIELDS) or not parts[-1].startswith(("http://", "https://")):
            continue  # blank line, header, or malformed
        row = DirectoryRow(*parts[: len(FIELDS)])
        rows.setdefault(row.url, row)
    return list(rows.values())


def fetch_directory(url: str, user_agent: str, client: httpx.Client | None = None) -> str:
    owns = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30)
    try:
        resp = client.get(url, headers={"User-Agent": user_agent})
        resp.raise_for_status()
        return resp.text
    finally:
        if owns:
            client.close()


def snapshot_frame(rows: list[DirectoryRow], snapshot_date: str) -> pl.DataFrame:
    data = [{"snapshot_date": snapshot_date, **r.__dict__} for r in rows]
    return pl.DataFrame(data, schema=DIRECTORY_SNAPSHOT)


def diff_snapshots(previous: list[DirectoryRow], current: list[DirectoryRow]) -> DirectoryDiff:
    old = {r.url: r for r in previous}
    new = {r.url: r for r in current}
    return DirectoryDiff(
        new=sorted(new.keys() - old.keys()),
        removed=sorted(old.keys() - new.keys()),
        changed=sorted(
            u
            for u in new.keys() & old.keys()
            if any(getattr(new[u], f) != getattr(old[u], f) for f in COMPARED)
        ),
    )


def _person_id(url: str) -> str:
    return "p_" + hashlib.sha1(url.encode()).hexdigest()[:12]


def _place_key(name: str, city: str) -> tuple[str, str]:
    return (name.casefold(), city.casefold())


def update_people(
    people: pl.DataFrame | None, rows: list[DirectoryRow], today: str
) -> tuple[pl.DataFrame, DirectoryDiff]:
    """Apply a directory snapshot to the ``people`` SCD2 table.

    A new URL keeps an existing ``person_id`` when name + city match a current or previously
    seen person (people change domains and own several URLs); otherwise it gets a new id.
    Attribute changes close the old interval and open a new one. URLs missing from the
    directory are closed, not deleted.
    """
    people = people if people is not None else empty(PEOPLE)
    current = {r.url: r for r in rows}
    open_rows = people.filter(pl.col("valid_to").is_null())
    open_by_url = {r["url"]: r for r in open_rows.iter_rows(named=True)}

    by_place: dict[tuple[str, str], str] = {}
    by_url: dict[str, str] = {}
    for r in people.iter_rows(named=True):  # later rows win, so the most recent mapping is used
        by_place[_place_key(r["name"], r["city"])] = r["person_id"]
        by_url[r["url"]] = r["person_id"]

    close: set[str] = set()
    additions: list[dict[str, str | None]] = []

    for url, p in open_by_url.items():
        row = current.get(url)
        if row is None:
            close.add(url)
        elif any(getattr(row, f) != p[f] for f in COMPARED):
            close.add(url)
            additions.append(_person_row(p["person_id"], row, today))

    for url, row in current.items():
        if url in open_by_url:
            continue
        pid = by_url.get(url) or by_place.get(_place_key(row.name, row.city)) or _person_id(url)
        additions.append(_person_row(pid, row, today))
        by_place[_place_key(row.name, row.city)] = pid

    if close:
        people = people.with_columns(
            pl.when(pl.col("url").is_in(close) & pl.col("valid_to").is_null())
            .then(pl.lit(today))
            .otherwise(pl.col("valid_to"))
            .alias("valid_to")
        )
    if additions:
        people = pl.concat([people, pl.DataFrame(additions, schema=PEOPLE)])

    previous_urls = set(open_by_url)
    diff = DirectoryDiff(
        new=sorted(current.keys() - previous_urls),
        changed=sorted(u for u in close if u in current),
        removed=sorted(u for u in close if u not in current),
    )
    return people, diff


def _person_row(pid: str, row: DirectoryRow, today: str) -> dict[str, str | None]:
    return {
        "person_id": pid,
        "name": row.name,
        "url": row.url,
        "city": row.city,
        "state": row.state,
        "country": row.country,
        "valid_from": today,
        "valid_to": None,
    }


def run_directory(
    dataset: Storage, text: str, today: date | None = None
) -> tuple[DirectoryDiff, int]:
    """Snapshot ``text``, update ``people``; return (diff, number of rows)."""
    day = (today or date.today()).isoformat()
    rows = parse_directory(text)
    if not rows:
        raise ValueError("directory parsed to zero rows; refusing to overwrite state")
    dataset.write_table(f"directory_snapshots/{day}", snapshot_frame(rows, day))
    people, diff = update_people(dataset.read_table("people"), rows, day)
    dataset.write_table("people", people)
    log.info(
        "directory: %d rows, %d new, %d changed, %d removed",
        len(rows),
        len(diff.new),
        len(diff.changed),
        len(diff.removed),
    )
    return diff, len(rows)
