"""Directory -> crawl -> extract against synthetic data and a mock HTTP transport."""

import asyncio
import time
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from ntn.crawl import CrawlReport, is_denied, run_crawl
from ntn.directory import diff_snapshots, parse_directory, run_directory
from ntn.extract import is_meaningful_change, normalize, parse_stated_date, run_extract
from ntn.storage import Storage

UA = "TestAgent/0.1"
PARAGRAPH = (
    "I am writing a book about gardens and learning to bake sourdough. "
    "This season I am mostly working on the second chapter and walking every morning. "
)


def tsv(*rows: tuple[str, str, str]) -> str:
    # name, city, url; remaining fields are filled in
    return "\n".join(f"{n}\t{c}\tState\tUS\t2020-01\t2026-01\t2026-10\t{u}" for n, c, u in rows)


def page(body: str, extra: str = "") -> bytes:
    return (
        f"<html><head>{extra}</head><body><article><p>{body}</p></article></body></html>".encode()
    )


def storages(tmp_path: Path) -> tuple[Storage, Storage]:
    return Storage(str(tmp_path / "ds")), Storage(str(tmp_path / "bucket"))


def test_parse_directory_skips_malformed_and_duplicates() -> None:
    text = "header\n" + tsv(("Ann", "Lisbon", "https://a.test/now")) + "\nbad\tline\n"
    text += "\n" + tsv(("Ann", "Lisbon", "https://a.test/now"))
    assert [r.url for r in parse_directory(text)] == ["https://a.test/now"]


def test_diff_snapshots() -> None:
    old = parse_directory(tsv(("A", "X", "https://a.test"), ("B", "Y", "https://b.test")))
    new = parse_directory(tsv(("A", "Z", "https://a.test"), ("C", "Y", "https://c.test")))
    d = diff_snapshots(old, new)
    assert (d.new, d.changed, d.removed) == (
        ["https://c.test"],
        ["https://a.test"],
        ["https://b.test"],
    )


def test_people_scd2_and_person_id_continuity(tmp_path: Path) -> None:
    ds, _ = storages(tmp_path)
    run_directory(ds, tsv(("Ann", "Lisbon", "https://old.test/now")), date(2026, 10, 1))
    # Ann changes domain: same name + city keeps her person_id; old URL is closed.
    diff, _ = run_directory(ds, tsv(("Ann", "Lisbon", "https://new.test/now")), date(2026, 10, 8))
    people = ds.read_table("people")
    assert people is not None
    assert people["person_id"].n_unique() == 1
    assert diff.removed == ["https://old.test/now"] and diff.new == ["https://new.test/now"]
    closed = people.filter(pl.col("url") == "https://old.test/now")
    assert closed["valid_to"].to_list() == ["2026-10-08"]
    # re-running the same snapshot is idempotent
    run_directory(ds, tsv(("Ann", "Lisbon", "https://new.test/now")), date(2026, 10, 9))
    assert ds.read_table("people").height == 2  # type: ignore[union-attr]


def test_storage_roundtrip(tmp_path: Path) -> None:
    s = Storage(str(tmp_path))
    s.write_bytes("a/b/c.bin", b"x")
    s.write_table("t/one", pl.DataFrame({"k": [1]}))
    s.write_table("t/two", pl.DataFrame({"k": [2]}))
    assert s.read_bytes("a/b/c.bin") == b"x"
    assert s.read_tables("t") is not None and s.read_tables("t").height == 2  # type: ignore[union-attr]
    assert s.read_table("missing") is None
    s.delete("a/b/c.bin")
    assert not s.exists("a/b/c.bin")


def test_normalize_and_stated_date() -> None:
    text = "Hello world.\n\nLast updated: March 3, 2026\n\n1,234 visitors\n\nBye."
    assert "updated" not in normalize(text).lower()
    assert "visitors" not in normalize(text)
    assert parse_stated_date(text) == "2026-03-03"
    assert not is_meaningful_change("a\nb\nc", "a\nb\nc")
    assert is_meaningful_change("a\nb\nc", "a\nx\ny")


def test_denylist_matches_url_and_host() -> None:
    assert is_denied("https://x.test/now#a", {"https://x.test/now"})
    assert is_denied("https://sub.x.test/now", {"x.test"})
    assert not is_denied("https://y.test/now", {"x.test"})


def test_end_to_end(tmp_path: Path) -> None:
    ds, bucket = storages(tmp_path)
    state = {"body": page(PARAGRAPH * 3)}

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if path == "/robots.txt":
            if host == "blocked.test":
                return httpx.Response(200, text="User-agent: *\nDisallow: /")
            return httpx.Response(404)
        if host == "ok.test":
            if request.headers.get("if-none-match") == '"v1"':
                return httpx.Response(304, headers={"etag": '"v1"'})
            return httpx.Response(
                200, content=state["body"], headers={"etag": '"v1"', "content-type": "text/html"}
            )
        if host == "noarchive.test":
            return httpx.Response(
                200, content=page("x" * 300, '<meta name="robots" content="noarchive">')
            )
        if host == "frag.test":
            nav, now = "nav " * 80, PARAGRAPH * 3
            body = (
                f'<html><body><div id="other"><p>{nav}</p></div>'
                f'<div id="now"><p>{now}</p></div></body></html>'
            )
            return httpx.Response(200, content=body.encode())
        if host == "blocked.test":
            raise AssertionError("robots.txt must prevent this request")
        return httpx.Response(500)

    run_directory(
        ds,
        tsv(
            ("Ok", "A", "https://ok.test/now"),
            ("Blocked", "B", "https://blocked.test/now"),
            ("NoArc", "C", "https://noarchive.test/now"),
            ("Frag", "D", "https://frag.test/#now"),
            ("Broken", "E", "https://broken.test/now"),
        ),
        date(2026, 10, 1),
    )

    def crawl() -> CrawlReport:
        return asyncio.run(
            run_crawl(
                ds, bucket, user_agent=UA, host_delay=0, retry_wait=0,
                transport=httpx.MockTransport(handler),
            )
        )  # fmt: skip

    r1 = crawl()
    assert (r1.attempted, r1.ok, r1.failed, r1.skipped) == (3, 2, 1, 2)
    assert r1.errors == {"http_500": 1, "noarchive": 1, "robots": 1}
    fetches = ds.read_tables("fetches")
    assert fetches is not None
    by_url = {r["url"]: r for r in fetches.iter_rows(named=True)}
    assert by_url["https://blocked.test/now"]["error"] == "robots"
    assert by_url["https://noarchive.test/now"]["error"] == "noarchive"
    assert by_url["https://broken.test/now"]["error"] == "http_500"
    assert by_url["https://ok.test/now"]["status"] == 200
    assert bucket.exists(by_url["https://ok.test/now"]["raw_object_key"] + ".html")
    assert not any("noarchive" in k for k in bucket.list("raw"))  # nothing stored for noarchive

    # Second crawl: conditional GET -> 304, no new raw object.
    raw_before = bucket.list("raw")
    crawl()
    assert bucket.list("raw") == raw_before
    again = ds.read_tables("fetches")
    assert again is not None
    assert again.filter(pl.col("url") == "https://ok.test/now")["status"].to_list().count(304) == 1

    # Extract: one version per good page, fragment selects the element.
    report = run_extract(ds, bucket)
    assert report.new_versions == 2 and report.errors == 0
    versions = ds.read_table("versions")
    assert versions is not None
    texts = {r["url"]: r["text_md"] for r in versions.iter_rows(named=True)}
    assert "sourdough" in texts["https://ok.test/now"]
    assert "sourdough" in texts["https://frag.test/#now"]
    assert "nav nav" not in texts["https://frag.test/#now"]
    assert versions["person_id"].null_count() == 0

    # Idempotent; the 304 advanced last_seen without making a version.
    assert run_extract(ds, bucket).new_versions == 0
    assert ds.read_table("versions").height == 2  # type: ignore[union-attr]

    # Page changes meaningfully -> exactly one new version for it.
    state["body"] = page("Completely different words now. " * 30)

    def changed(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/robots.txt" and request.url.host == "ok.test":
            return httpx.Response(200, content=state["body"], headers={"etag": '"v2"'})
        return handler(request)

    asyncio.run(
        run_crawl(
            ds, bucket, user_agent=UA, host_delay=0, retry_wait=0,
            transport=httpx.MockTransport(changed),
        )
    )  # fmt: skip
    assert run_extract(ds, bucket).new_versions == 1
    ok_versions = ds.read_table("versions").filter(pl.col("url") == "https://ok.test/now")  # type: ignore[union-attr]
    assert ok_versions.height == 2


def test_circuit_breaker_marks_run_unhealthy(tmp_path: Path) -> None:
    ds, bucket = storages(tmp_path)
    run_directory(ds, tsv(("A", "X", "https://a.test/now"), ("B", "Y", "https://b.test/now")))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404 if request.url.path == "/robots.txt" else 503)

    report = asyncio.run(
        run_crawl(ds, bucket, user_agent=UA, host_delay=0, retry_wait=0,
                  transport=httpx.MockTransport(handler))
    )  # fmt: skip
    assert not report.healthy and report.failed == 2


def test_cache_comment_stores_a_new_raw_but_not_a_new_version(tmp_path: Path) -> None:
    ds, bucket = storages(tmp_path)
    url = "https://wp.test/now"
    comments = [
        "<!-- Dynamic page generated in 0.085 seconds. -->"
        "<!-- Cached page generated by WP-Super-Cache on 2026-10-04 05:57:27 -->",
        "<!-- Dynamic page generated in 0.081 seconds. -->"
        "<!-- Cached page generated by WP-Super-Cache on 2026-10-04 06:33:38 -->",
    ]
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            content=page(PARAGRAPH * 3, comments[seen["n"]]),
            headers={"content-type": "text/html"},
        )

    run_directory(ds, tsv(("Ada", "X", url)))

    def crawl() -> None:
        asyncio.run(
            run_crawl(
                ds,
                bucket,
                user_agent=UA,
                host_delay=0,
                retry_wait=0,
                transport=httpx.MockTransport(handler),
            )
        )

    crawl()
    seen["n"] = 1
    time.sleep(1.1)  # run ids are per second; the second crawl must not overwrite the first
    crawl()
    htmls = [k for k in bucket.list("raw") if k.endswith(".html")]
    assert len(htmls) == 2
    report = run_extract(ds, bucket)
    assert report.new_versions == 1 and report.unchanged == 1
    assert ds.read_table("versions").height == 1  # type: ignore[union-attr]
