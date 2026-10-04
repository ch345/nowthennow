import typer

from ntn import __version__

app = typer.Typer(help="now then now data pipeline.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Crawl, index, and enrich /now pages."""


@app.command()
def version() -> None:
    """Print the pipeline version."""
    typer.echo(__version__)
