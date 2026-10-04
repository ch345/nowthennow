"""Enrichment against a fake OpenAI-compatible client."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from ntn import enrich
from ntn.config import Settings
from ntn.storage import Storage
from ntn.tables import VERSIONS

TEXT = "I am writing a novel about grief.\n\nThis autumn I am learning to bake sourdough bread."


def test_valid_quotes_keeps_only_verbatim_substrings() -> None:
    got = enrich.valid_quotes(["writing  a novel\nabout grief", "I am climbing Everest", ""], TEXT)
    assert got == ["writing a novel about grief"]


class FakeClient:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        page = kwargs["messages"][1]["content"]
        if self.fail_on and self.fail_on in page:
            raise RuntimeError("boom")
        payload = {
            "language": "en",
            "richness": 0.7,
            "stated_location": "",
            "focuses": [
                {
                    "domain": "creative",
                    "summary": "writing a novel about grief",
                    "tags": ["writing", "grief"],
                    "status": "ongoing",
                    "sensitive": False,
                }
            ],
            "life_transitions": [],
            "values": ["craft"],
            "tone": "reflective",
            "quotes": ["I am writing a novel about grief.", "invented quote"],
        }
        msg = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def seed(dataset: Storage, texts: list[str]) -> None:
    rows = [
        {
            "version_id": f"v{i}",
            "person_id": f"p{i}",
            "url": f"https://e{i}.example",
            "first_seen": "2026-10-01",
            "last_seen": "2026-10-01",
            "source": "crawl",
            "archive_url": None,
            "stated_date": None,
            "text_md": t,
            "language": None,
            "richness": None,
            "content_hash": "h",
            "text_hash": "t",
            "needs_js": False,
        }
        for i, t in enumerate(texts)
    ]
    dataset.write_table("versions", pl.DataFrame(rows, schema=VERSIONS))


def test_run_enrich_is_incremental_and_tolerates_failures(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Storage(str(tmp_path))
    seed(dataset, [TEXT, TEXT + " second", "BADPAGE " + TEXT])
    client = FakeClient(fail_on="BADPAGE")

    def fake_client(settings: Settings, *, is_async: bool) -> FakeClient:
        return client

    monkeypatch.setattr(enrich, "_client", fake_client)
    settings = Settings(llm_api_key="x")

    r = asyncio.run(enrich.run_enrich(dataset, settings))
    assert (r.enriched, r.failed, r.focuses) == (2, 1, 2)

    focuses = dataset.read_table("focuses")
    assert focuses is not None
    assert sorted(focuses["focus_id"].to_list()) == ["v0:0", "v1:0"]
    extractions = dataset.read_table("extractions")
    assert extractions is not None
    assert json.loads(extractions["json"][0])["quotes"] == ["I am writing a novel about grief."]

    # Second run only retries the failure.
    before = client.calls
    asyncio.run(enrich.run_enrich(dataset, settings))
    assert client.calls == before + 1


def test_schema_and_prompt_follow_config() -> None:
    schema = enrich.enrichment_schema()
    domains = schema["$defs"]["Focus"]["properties"]["domain"]["enum"]
    assert "other" in domains
    assert all(d in enrich.system_prompt() for d in domains)


def test_config_rejects_unknown_keys_and_missing_catch_all() -> None:
    from pydantic import ValidationError

    from ntn.config import Tuning

    with pytest.raises(ValidationError):
        Tuning.model_validate({"enrich": {"domain": ["work"]}})  # typo
    with pytest.raises(ValidationError):
        Tuning.model_validate({"enrich": {"domains": ["work"]}})  # no "other"


def test_committed_ntn_toml_matches_defaults() -> None:
    import tomllib
    from pathlib import Path

    from ntn.config import Tuning

    loaded = Tuning.model_validate(
        tomllib.loads((Path(__file__).parent.parent / "ntn.toml").read_text())
    )
    assert loaded == Tuning()
