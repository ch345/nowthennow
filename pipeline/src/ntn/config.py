"""Runtime configuration, read from environment variables.

Design decisions (domains, thresholds, models) live in ``ntn.toml``; see ``tuning()``.

Storage locations are fsspec URLs so the backend can be swapped by configuration:

- ``NTN_DATASET_URL``: tables (Parquet). ``hf://datasets/<owner>/<name>`` or a local path.
- ``NTN_BUCKET_URL``: raw HTML. ``hf://buckets/<owner>/<name>``, ``s3://...`` or a local path.
- ``HF_TOKEN``: read by ``huggingface_hub`` for ``hf://`` URLs.
- ``NTN_LLM_BASE_URL`` / ``NTN_LLM_MODEL``: override the ``[llm]`` section of ntn.toml (any
  OpenAI-compatible endpoint). The key is ``NTN_LLM_API_KEY``, falling back to ``PRIME_API_KEY``.
"""

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DIRECTORY_URL = "https://nownownow.com/nownownow.txt"
LLM_BASE_URL = "https://api.pinference.ai/api/v1"
LLM_MODEL = "openai/gpt-4.1-mini"
USER_AGENT = "NowThenNow/0.1 (+https://ch345.github.io/nowthennow/about)"


@dataclass(frozen=True)
class Settings:
    dataset_url: str = "./data/dataset"
    bucket_url: str = "./data/bucket"
    directory_url: str = DIRECTORY_URL
    user_agent: str = USER_AGENT
    llm_base_url: str = ""  # empty: use ntn.toml
    llm_model: str = ""
    llm_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            dataset_url=os.environ.get("NTN_DATASET_URL", defaults.dataset_url),
            bucket_url=os.environ.get("NTN_BUCKET_URL", defaults.bucket_url),
            directory_url=os.environ.get("NTN_DIRECTORY_URL", defaults.directory_url),
            llm_base_url=os.environ.get("NTN_LLM_BASE_URL") or tuning().llm.base_url,
            llm_model=os.environ.get("NTN_LLM_MODEL") or tuning().llm.model,
            llm_api_key=os.environ.get("NTN_LLM_API_KEY") or os.environ.get("PRIME_API_KEY"),
        )


# -- tuning: the project's design decisions, kept in one file (ntn.toml) ------------------------


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in ntn.toml is an error, not a no-op


class LlmCfg(_Section):
    base_url: str = LLM_BASE_URL
    model: str = LLM_MODEL


class CrawlCfg(_Section):
    concurrency: int = 20  # max in-flight requests overall
    host_delay: float = 1.0  # seconds between requests to one host
    max_bytes: int = 5_000_000
    failure_threshold: float = 0.30  # more failures than this marks the run unhealthy


class ExtractCfg(_Section):
    min_text_chars: int = 200  # shorter text is flagged as probably JavaScript-rendered
    meaningful_ratio: float = 0.98  # line similarity at or above this is "the same version"


class EnrichCfg(_Section):
    domains: list[str] = Field(
        default=[
            "work",
            "family",
            "health",
            "learning",
            "creative",
            "place",
            "community",
            "rest",
            "other",
        ],
        min_length=1,
    )
    statuses: list[str] = Field(default=["starting", "ongoing", "ending"], min_length=1)
    life_transitions: list[str] = [
        "moved",
        "new_child",
        "left_job",
        "new_job",
        "sabbatical",
        "loss",
        "illness",
        "other",
    ]
    sensitive_topics: list[str] = [
        "health",
        "sexuality",
        "religion",
        "politics",
        "finances",
    ]  # focuses about these are flagged `sensitive` and kept out of embeddings and the map
    prompt_version: str = "1"  # bump to re-enrich every page
    max_text_chars: int = 12_000  # page text sent to the model
    max_focuses: int = 5
    max_quotes: int = 3
    summary_words: int = 25
    concurrency: int = 8
    checkpoint_every: int = 25


class EmbedCfg(_Section):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 128


class ClusterCfg(_Section):
    min_cluster_size: int = 8  # focuses per topic
    min_topic_pages: int = 3  # a topic needs this many different pages
    merge_cosine: float = 0.85  # topics closer than this are merged
    small_threshold: int = 100  # fewer focuses than this: cluster on raw embeddings, not UMAP
    seed: int = 42
    label_terms: int = 3  # words in a keyword-fallback topic label
    label_max_page_share: float = 0.4  # keyword labels ignore words on more pages than this
    topic_sample_size: int = 10  # focus summaries shown to the LLM when naming a topic
    llm_names: bool = True
    excerpt_chars: int = 280


class Tuning(_Section):
    llm: LlmCfg = LlmCfg()
    crawl: CrawlCfg = CrawlCfg()
    extract: ExtractCfg = ExtractCfg()
    enrich: EnrichCfg = EnrichCfg()
    embed: EmbedCfg = EmbedCfg()
    cluster: ClusterCfg = ClusterCfg()

    @model_validator(mode="after")
    def _check(self) -> "Tuning":
        if "other" not in self.enrich.domains:
            raise ValueError("enrich.domains must include 'other' as a catch-all")
        return self


@lru_cache
def tuning() -> Tuning:
    """Load ``ntn.toml``: ``$NTN_CONFIG``, else ``./ntn.toml``, else built-in defaults."""
    path = Path(os.environ.get("NTN_CONFIG", "ntn.toml"))
    if not path.exists():
        if "NTN_CONFIG" in os.environ:
            raise FileNotFoundError(f"NTN_CONFIG={path} does not exist")
        return Tuning()
    return Tuning.model_validate(tomllib.loads(path.read_text()))
