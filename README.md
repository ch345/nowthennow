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
| `pipeline/`   | Python package (`ntn`): crawl, extract, enrich, embed, cluster  |
| `web/`        | React + TypeScript site, deployed to GitHub Pages               |
| `.github/`    | CI, scheduled crawl, and deploy workflows                       |

Archived page content is stored privately on Hugging Face, never in this repository.

## Status

Early proof of concept. The pipeline runs locally from crawl to topic clusters, and the web app
draws them as a scatter plot from a local, git-ignored data file. The deployed site does not show
data yet: publishing public-safe data at deploy time is not built.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Node 22+.

```sh
# pipeline
cd pipeline && uv sync --extra ml && uv run pytest   # see pipeline/README.md

# web (shows data if pipeline/ has produced web/public/data/points.json)
cd web && npm install && npm run dev

# git hooks
uvx pre-commit install
```

## Removal requests

If you are listed on nownownow.com and want your page excluded from this archive, email me (cohuang@mit.edu).
Removal deletes all stored versions, not just the published ones.
