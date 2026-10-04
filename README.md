# now then now

An atlas of [/now pages](https://nownownow.com/about): what people around the world say they are
focused on at this point in their lives, archived over time.

A /now page is written in the present and read later. Every one you read is a message from a
moment that has already passed. This project crawls the pages listed on
[nownownow.com](https://nownownow.com), keeps each distinct version, and maps them by place, by
time, and by what people care about.

## Repository layout

| Path          | What it is                                                      |
| ------------- | --------------------------------------------------------------- |
| `pipeline/`   | Python package (`ntn`): crawler, extraction, enrichment         |
| `web/`        | React + TypeScript site, deployed to GitHub Pages               |
| `.github/`    | CI, scheduled crawl, and deploy workflows                       |

Archived page content is stored privately on Hugging Face, never in this repository.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Node 22+.

```sh
# pipeline
cd pipeline && uv sync && uv run pytest

# web
cd web && npm install && npm run dev

# git hooks
uvx pre-commit install
```

## Removal requests

If you are listed on nownownow.com and want your page excluded from this archive, email me (cohuang@mit.edu).
Removal deletes all stored versions, not just the published ones.
