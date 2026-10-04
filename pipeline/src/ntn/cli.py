import asyncio
import logging
import os
from typing import Annotated

import typer

from ntn import __version__
from ntn.config import Settings, tuning
from ntn.storage import Storage

app = typer.Typer(help="now then now data pipeline.", no_args_is_help=True)

Limit = Annotated[int | None, typer.Option(help="Only process this many URLs (for development).")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Crawl, index, and enrich /now pages."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    # httpx2 is what huggingface_hub uses; both log every request at INFO, including the URL.
    for name in ("httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Read at huggingface_hub import, which happens inside the commands.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@app.command()
def version() -> None:
    """Print the pipeline version."""
    typer.echo(__version__)


def _storages() -> tuple[Settings, Storage, Storage]:
    settings = Settings.from_env()
    return settings, Storage(settings.dataset_url), Storage(settings.bucket_url)


@app.command("init-hf")
def init_hf(
    dataset: Annotated[str, typer.Argument(help="Dataset repo id, e.g. user/nowthennow-data")],
    bucket: Annotated[str, typer.Argument(help="Bucket id, e.g. user/nowthennow-raw")],
) -> None:
    """Create the private Hugging Face dataset and bucket (idempotent)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    api.create_bucket(bucket, private=True, exist_ok=True)
    typer.echo(f"export NTN_DATASET_URL=hf://datasets/{dataset}")
    typer.echo(f"export NTN_BUCKET_URL=hf://buckets/{bucket}")


@app.command()
def directory() -> None:
    """Fetch nownownow.txt, snapshot it, and update the people table."""
    from ntn.directory import fetch_directory, run_directory

    settings, dataset, _ = _storages()
    text = fetch_directory(settings.directory_url, settings.user_agent)
    diff, total = run_directory(dataset, text)
    typer.echo(
        f"{total} rows: {len(diff.new)} new, {len(diff.changed)} changed, "
        f"{len(diff.removed)} removed"
    )


@app.command()
def crawl(
    limit: Limit = None,
    concurrency: Annotated[int | None, typer.Option(help="Max in-flight requests overall.")] = None,
    host_delay: Annotated[
        float | None, typer.Option(help="Seconds between requests to a host.")
    ] = None,
) -> None:
    """Fetch current /now pages; store raw HTML to the bucket when it changed."""
    from ntn.crawl import SKIP_ERRORS, run_crawl

    settings, dataset, bucket = _storages()
    report = asyncio.run(
        run_crawl(
            dataset,
            bucket,
            user_agent=settings.user_agent,
            limit=limit,
            concurrency=concurrency or tuning().crawl.concurrency,
            host_delay=tuning().crawl.host_delay if host_delay is None else host_delay,
        )
    )
    typer.echo(
        f"run {report.run_id}: {report.attempted} fetched, {report.ok} ok, "
        f"{report.not_modified} not modified, {report.failed} failed, {report.skipped} skipped"
    )
    if report.errors:
        skips = ", ".join(
            f"{k}={report.errors[k]}" for k in sorted(report.errors) if k in SKIP_ERRORS
        )
        fails = ", ".join(
            f"{k}={report.errors[k]}" for k in sorted(report.errors) if k not in SKIP_ERRORS
        )
        parts = []
        if skips:
            parts.append(f"skipped {skips}")
        if fails:
            parts.append(f"failed {fails}")
        typer.echo("reasons: " + "; ".join(parts))
    if not report.healthy:
        typer.echo(f"UNHEALTHY: {report.failure_rate:.0%} of fetches failed", err=True)
        raise typer.Exit(2)


@app.command()
def extract() -> None:
    """Turn raw HTML into versioned markdown."""
    from ntn.extract import run_extract

    _, dataset, bucket = _storages()
    r = run_extract(dataset, bucket)
    typer.echo(
        f"{r.examined} fetches examined: {r.new_versions} new versions, {r.unchanged} unchanged, "
        f"{r.needs_js} need JS rendering, {r.errors} errors"
    )


@app.command()
def enrich(
    limit: Limit = None,
    concurrency: Annotated[int | None, typer.Option(help="Max in-flight LLM requests.")] = None,
) -> None:
    """Extract focuses, tags and quotes from each page version with an LLM."""
    from ntn.enrich import run_enrich

    settings, dataset, _ = _storages()
    r = asyncio.run(run_enrich(dataset, settings, limit=limit, concurrency=concurrency))
    typer.echo(
        f"{r.candidates} versions to enrich: {r.enriched} ok, {r.failed} failed, "
        f"{r.focuses} new focuses"
    )
    if r.failed:
        raise typer.Exit(1)


@app.command()
def embed() -> None:
    """Embed each focus summary (needs the `ml` extra)."""
    from ntn.embed import run_embed

    _, dataset, _ = _storages()
    r = run_embed(dataset)
    typer.echo(f"{r.candidates} focuses, {r.embedded} newly embedded")


@app.command()
def cluster(
    export: Annotated[
        str, typer.Option(help="Write the web app's data file here.")
    ] = "../web/public/data/points.json",
    min_cluster_size: Annotated[
        int | None, typer.Option(help="Minimum focuses per topic (HDBSCAN).")
    ] = None,
    llm_names: Annotated[bool | None, typer.Option(help="Name topics with the LLM.")] = None,
) -> None:
    """Lay out pages (UMAP) and find topics (HDBSCAN over focuses); export for the web app."""
    from ntn.cluster import run_cluster, write_export

    settings, dataset, _ = _storages()
    report, data = run_cluster(dataset, settings, min_cluster_size, llm_names=llm_names)
    write_export(data, export)
    typer.echo(
        f"run {report.run_id}: {report.points} points, {report.clusters} clusters, "
        f"{report.noise} unclustered -> {export}"
    )
