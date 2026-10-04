"""Polars schemas for the Parquet tables (see docs/ARCHITECTURE.md, "Data model")."""

import polars as pl

DIRECTORY_SNAPSHOT = pl.Schema(
    {
        "snapshot_date": pl.String,
        "name": pl.String,
        "city": pl.String,
        "state": pl.String,
        "country": pl.String,
        "created": pl.String,
        "updated": pl.String,
        "checked": pl.String,
        "url": pl.String,
    }
)

# Type 2 slowly changing dimension: one row per (person_id, url, attributes) interval.
PEOPLE = pl.Schema(
    {
        "person_id": pl.String,
        "name": pl.String,
        "url": pl.String,
        "city": pl.String,
        "state": pl.String,
        "country": pl.String,
        "valid_from": pl.String,
        "valid_to": pl.String,  # null while current
    }
)

FETCHES = pl.Schema(
    {
        "run_id": pl.String,
        "url": pl.String,
        "fetched_at": pl.String,
        "status": pl.Int64,  # null when no response was received
        "final_url": pl.String,
        "etag": pl.String,
        "last_modified": pl.String,
        "content_hash": pl.String,
        "bytes": pl.Int64,
        "error": pl.String,
        "raw_object_key": pl.String,
    }
)

VERSIONS = pl.Schema(
    {
        "version_id": pl.String,
        "person_id": pl.String,
        "url": pl.String,
        "first_seen": pl.String,
        "last_seen": pl.String,
        "source": pl.String,  # crawl | wayback
        "archive_url": pl.String,
        "stated_date": pl.String,
        "text_md": pl.String,
        "language": pl.String,
        "richness": pl.Float64,
        "content_hash": pl.String,  # hash of the raw bytes behind the latest fetch of this version
        "text_hash": pl.String,  # hash of the normalized text
        "needs_js": pl.Boolean,  # extraction near-empty: probably JavaScript-rendered
    }
)


EXTRACTIONS = pl.Schema(
    {
        "version_id": pl.String,
        "model": pl.String,
        "prompt_version": pl.String,
        "schema_version": pl.String,
        "json": pl.String,
        "created_at": pl.String,
    }
)

FOCUSES = pl.Schema(
    {
        "focus_id": pl.String,  # <version_id>:<index>
        "version_id": pl.String,
        "domain": pl.String,
        "summary": pl.String,
        "tags": pl.List(pl.String),
        "status": pl.String,
        "sensitive": pl.Boolean,
    }
)

EMBEDDINGS = pl.Schema(
    {
        "id": pl.String,  # version_id for kind == "page"
        "kind": pl.String,  # page | focus
        "model": pl.String,
        "model_revision": pl.String,
        "vector": pl.List(pl.Float32),
    }
)

LAYOUTS = pl.Schema(
    {
        "run_id": pl.String,
        "id": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "cluster_id": pl.Int64,  # -1 = noise
        "cluster_label": pl.String,
    }
)


def empty(schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
