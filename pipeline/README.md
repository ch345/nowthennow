# ntn pipeline

Python package that crawls /now pages, extracts and versions their text, enriches it with an
LLM, and clusters what people are focused on. Stages, in order: `directory`, `crawl`, `extract`,
`enrich`, `embed`, `cluster`.

```sh
uv sync --extra ml    # `ml` (onnxruntime, numpy, scikit-learn, umap-learn) is needed by embed/cluster and by pyright
uv run ntn --help
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

## Running the stages

Storage locations are fsspec URLs (defaults are local, under the git-ignored `./data/`):

```sh
export HF_TOKEN=...                      # fine-grained, scoped to the one dataset + bucket
uv run ntn init-hf USER/nowthennow-data USER/nowthennow-raw   # once; prints the exports below
export NTN_DATASET_URL=hf://datasets/USER/nowthennow-data   # Parquet tables (private)
export NTN_BUCKET_URL=hf://buckets/USER/nowthennow-raw      # raw HTML (private)

uv run ntn directory          # snapshot nownownow.txt, update `people`
uv run ntn crawl --limit 20   # polite incremental fetch; exits 2 if >30% of fetches fail
uv run ntn extract            # raw HTML -> versioned markdown
```

Design decisions (enrichment vocabularies, thresholds, models, crawl politeness) live in
[`ntn.toml`](ntn.toml); secrets and storage locations stay in environment variables. Every key
is optional, typos are errors, and `NTN_CONFIG` points at an alternative file. After editing
`[enrich]` vocabularies, bump `prompt_version` to re-enrich.

A `denylist.txt` in the dataset (one URL or domain per line) excludes pages from crawling.

## Enrich, embed, cluster, and the map (proof of concept)

```sh
export PRIME_API_KEY=...           # or NTN_LLM_API_KEY; endpoint/model: [llm] in ntn.toml,
                                   # or NTN_LLM_BASE_URL / NTN_LLM_MODEL
uv run ntn enrich --limit 5        # try a few pages first; then without --limit
uv run ntn embed                   # focus summaries -> `embeddings` (MiniLM, ONNX)
uv run ntn cluster                 # -> `layouts` and ../web/public/data/points.json
cd ../web && npm run dev
```

- `enrich` writes `extractions` and `focuses` and is incremental per `prompt_version`. Quotes
  that are not verbatim are dropped. Pages flagged `needs_js` are skipped (no Playwright
  fallback yet).
- `embed` skips focuses flagged `sensitive`. The first run downloads the model from Hugging Face.
- `cluster` clusters focuses, so a page can belong to several topics; topic names come from the
  LLM (`--no-llm-names` for keyword labels). Options such as `--min-cluster-size` override
  `ntn.toml`. `points.json` contains names, excerpts and links, so it is git-ignored and not
  published.

Not built yet: `publish`, the scheduled workflow, backfill, removal, geocoding. See the TODO
list in `docs/ARCHITECTURE.md`.
