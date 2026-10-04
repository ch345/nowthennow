# ntn pipeline

Python package that crawls /now pages, extracts and versions their text, and produces the
datasets the web app reads.

```sh
uv sync
uv run ntn --help
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```
