"""Stage 5: embed the English focus summaries produced by ``ntn enrich``.

One vector per focus (not per page): a page that is about code and about reading has focuses in
both places. Page-level positions are derived later by averaging a page's focus vectors.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ntn.config import tuning
from ntn.storage import Storage
from ntn.tables import EMBEDDINGS, empty

log = logging.getLogger(__name__)


def latest_versions(dataset: Storage) -> pl.DataFrame:
    """The most recent usable version of each URL."""
    versions = dataset.read_table("versions")
    if versions is None or versions.is_empty():
        return pl.DataFrame()
    return (
        versions.filter(pl.col("text_md").is_not_null() & ~pl.col("needs_js").fill_null(False))
        .filter(pl.col("text_md").str.len_chars() > 0)
        .sort("last_seen")
        .unique(subset="url", keep="last", maintain_order=True)
    )


def encode(texts: list[str], model_name: str, batch_size: int = 16) -> tuple[Any, str]:
    """Mean-pooled, L2-normalized sentence embeddings via onnxruntime. Returns (matrix, revision).

    Uses onnxruntime + tokenizers directly: sentence-transformers and fastembed pin
    huggingface-hub below 2, which the storage layer needs.
    """
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    model_path = hf_hub_download(model_name, "onnx/model.onnx")
    tok_path = hf_hub_download(model_name, "tokenizer.json")
    revision = Path(model_path).parent.parent.name  # snapshots/<commit>/onnx/model.onnx
    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_truncation(max_length=tuning().embed.max_tokens)
    tokenizer.enable_padding()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    wanted = {i.name for i in session.get_inputs()}

    out: list[Any] = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer.encode_batch(texts[start : start + batch_size])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden: Any = session.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None))
    return np.vstack(out).astype(np.float32), revision


@dataclass
class EmbedReport:
    candidates: int  # non-sensitive focuses on current page versions
    embedded: int


def focus_text(summary: str, tags: list[str]) -> str:
    return f"{summary} ({', '.join(tags)})" if tags else summary


def run_embed(dataset: Storage, model_name: str | None = None) -> EmbedReport:
    model_name = model_name or tuning().embed.model
    latest = latest_versions(dataset)
    focuses = dataset.read_table("focuses")
    if latest.is_empty() or focuses is None:
        raise RuntimeError("no focuses; run `ntn enrich` first")
    current = focuses.filter(
        pl.col("version_id").is_in(latest["version_id"]) & ~pl.col("sensitive")
    )
    stored = dataset.read_table("embeddings")
    existing = empty(EMBEDDINGS) if stored is None else stored
    done = set(
        existing.filter((pl.col("model") == model_name) & (pl.col("kind") == "focus"))[
            "id"
        ].to_list()
    )
    todo = current.filter(~pl.col("focus_id").is_in(done))
    if todo.is_empty():
        return EmbedReport(current.height, 0)

    texts = [focus_text(r["summary"], r["tags"]) for r in todo.iter_rows(named=True)]
    vectors, revision = encode(texts, model_name)
    new = pl.DataFrame(
        {
            "id": todo["focus_id"].to_list(),
            "kind": ["focus"] * todo.height,
            "model": [model_name] * todo.height,
            "model_revision": [revision] * todo.height,
            "vector": vectors.tolist(),
        },
        schema=EMBEDDINGS,
    )
    dataset.write_table("embeddings", pl.concat([existing, new]))
    return EmbedReport(current.height, new.height)
