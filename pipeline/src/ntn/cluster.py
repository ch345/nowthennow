"""Stage 6: UMAP layout + HDBSCAN clusters over page embeddings.

Proof of concept: topic labels are distinctive terms (c-TF-IDF) rather than LLM names, and there
is no cross-run ID matching yet. Also exports a JSON file for the web app.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ntn.config import Settings, tuning
from ntn.crawl import new_run_id
from ntn.embed import focus_text, latest_versions
from ntn.enrich import name_topics
from ntn.storage import Storage
from ntn.tables import LAYOUTS

log = logging.getLogger(__name__)


@dataclass
class ClusterReport:
    run_id: str
    points: int
    clusters: int
    noise: int


def layout_2d(vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """UMAP coordinates for display; nearby points are similar pages."""
    from umap import UMAP

    n = len(vectors)
    if n < 4:  # UMAP cannot embed this few points
        return np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    return np.asarray(
        UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, n - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=tuning().cluster.seed,
            init="spectral" if n > 15 else "random",
        ).fit_transform(vectors)
    )


def cluster_ids(vectors: np.ndarray[Any, Any], min_cluster_size: int) -> np.ndarray[Any, Any]:
    """HDBSCAN cluster ids (-1 = noise).

    Small sets are clustered on the cosine geometry of the raw embeddings: a 5D UMAP of a few
    dozen points mostly invents structure. Larger sets use the usual UMAP(5D) -> HDBSCAN recipe.
    """
    from sklearn.cluster import HDBSCAN
    from umap import UMAP

    n = len(vectors)
    if n < max(4, min_cluster_size):
        return np.full(n, -1)
    features, metric = vectors, "cosine"
    if n >= tuning().cluster.small_threshold:
        features = UMAP(
            n_components=5,
            n_neighbors=max(2, min(15, n - 1)),
            min_dist=0.0,
            metric="cosine",
            random_state=tuning().cluster.seed,
        ).fit_transform(vectors)
        metric = "euclidean"
    return np.asarray(
        HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_cluster_size, metric=metric
        ).fit_predict(features)
    )


def merge_similar(
    vectors: np.ndarray[Any, Any], topic_of: np.ndarray[Any, Any], threshold: float | None = None
) -> np.ndarray[Any, Any]:
    """Merge topics whose centroids are nearly parallel (HDBSCAN often splits one theme in two)."""
    threshold = threshold if threshold is not None else tuning().cluster.merge_cosine
    topic_of = topic_of.copy()
    while True:
        ids = sorted(int(t) for t in set(topic_of.tolist()) if t >= 0)
        cents = {}
        for t in ids:
            c = vectors[topic_of == t].mean(axis=0)
            cents[t] = c / max(float(np.linalg.norm(c)), 1e-9)
        pairs = [
            (float(cents[a] @ cents[b]), a, b) for k, a in enumerate(ids) for b in ids[k + 1 :]
        ]
        if not pairs or max(pairs)[0] < threshold:
            return topic_of
        _, a, b = max(pairs)
        topic_of[topic_of == b] = a


def label_topics(
    texts: list[str], page_of: list[int], topic_of: Any, top: int | None = None
) -> dict[int, str]:
    """Name each topic from its passages with class-based TF-IDF.

    ``texts[i]`` is a passage on page ``page_of[i]`` assigned to topic ``topic_of[i]``. A term is a
    candidate only if it appears on at least two pages (drops names and one-offs) and on at most
    ~40% of pages (drops stopwords in any language and site boilerplate). Terms are then scored by
    how concentrated they are in a topic, and a label never repeats a word within itself.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    cfg = tuning().cluster
    top = top or cfg.label_terms

    topic_of = np.asarray(topic_of)
    ids = sorted(int(c) for c in set(topic_of.tolist()) if c >= 0)
    if not ids:
        return {}
    n_pages = len(set(page_of))
    pages = [
        " ".join(t for t, p in zip(texts, page_of, strict=True) if p == q) for q in set(page_of)
    ]
    token = r"(?u)\b[^\W\d_]{4,}\b"
    try:
        page_cv = CountVectorizer(
            token_pattern=token, ngram_range=(1, 2), lowercase=True, binary=True
        )
        page_counts: Any = page_cv.fit_transform(pages)
    except ValueError:
        return {i: f"theme {i}" for i in ids}
    page_df = np.asarray(page_counts.sum(axis=0)).ravel()
    keep = (page_df >= 2) & (page_df <= max(3, cfg.label_max_page_share * n_pages))
    vocab = page_cv.get_feature_names_out()[keep]
    if len(vocab) == 0:
        return {i: f"theme {i}" for i in ids}
    cv = CountVectorizer(token_pattern=token, ngram_range=(1, 2), vocabulary=vocab.tolist())
    docs = [" ".join(t for t, c in zip(texts, topic_of, strict=True) if c == i) for i in ids]
    tf = np.asarray(cv.transform(docs).todense(), dtype=float)
    tf = tf / np.clip(tf.sum(axis=1, keepdims=True), 1, None)
    total = tf.sum(axis=0)
    score = tf * np.log(1 + tf.shape[0] * tf.mean() / np.clip(total, 1e-12, None))
    out: dict[int, str] = {}
    used: set[str] = set()  # words in other topics' labels, so labels don't overlap
    for row, i in enumerate(ids):
        picked: list[str] = []
        words: set[str] = set()
        for j in np.argsort(score[row])[::-1]:
            if score[row, j] <= 0 or len(picked) == top:
                break
            term = str(vocab[j])
            if (words | used) & set(term.split()):
                continue
            picked.append(term)
            words |= set(term.split())
        used |= words
        out[i] = ", ".join(picked) or f"theme {i}"
    return out


def _excerpt(text: str) -> str:
    plain = re.sub(r"[#*_>`\[\]]|\(http[^)]*\)", "", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[: tuning().cluster.excerpt_chars] + (
        "…" if len(plain) > tuning().cluster.excerpt_chars else ""
    )


def run_cluster(
    dataset: Storage,
    settings: Settings,
    min_cluster_size: int | None = None,
    model: str | None = None,
    llm_names: bool | None = None,
) -> tuple[ClusterReport, dict[str, Any]]:
    """Pages get a position (similarity) and a weight on each topic (multi-membership).

    Topics come from clustering *focuses*, not pages, so a page with a coding focus and a reading
    focus belongs to both topics. A page's weight on a topic is its share of focuses there.
    """
    cfg = tuning().cluster
    min_cluster_size = min_cluster_size or cfg.min_cluster_size
    model = model or tuning().embed.model
    llm_names = cfg.llm_names if llm_names is None else llm_names
    emb, foc = dataset.read_table("embeddings"), dataset.read_table("focuses")
    latest = latest_versions(dataset)
    if emb is None or foc is None or latest.is_empty():
        raise RuntimeError("no embedded focuses; run `ntn enrich` and `ntn embed` first")
    emb = emb.filter((pl.col("model") == model) & (pl.col("kind") == "focus"))
    foc = foc.join(emb.select(pl.col("id").alias("focus_id"), "vector"), on="focus_id")
    pages = latest.filter(pl.col("version_id").is_in(foc["version_id"].unique()))
    extractions = dataset.read_table("extractions")
    quotes: dict[str, str] = {}
    if extractions is not None:
        for r in extractions.iter_rows(named=True):
            found = json.loads(r["json"]).get("quotes") or []
            if found:
                quotes[r["version_id"]] = found[0]
    people = dataset.read_table("people")
    if people is not None:
        current = (
            people.filter(pl.col("valid_to").is_null())
            .unique(subset="url", keep="last")
            .select("url", "name", "city", "country")
        )
        pages = pages.join(current, on="url", how="left")
    else:
        pages = pages.with_columns(
            pl.lit(None, pl.String).alias(c) for c in ("name", "city", "country")
        )
    pages = pages.sort("url")
    if pages.is_empty():
        raise RuntimeError("no pages with embedded focuses")
    n_pages = pages.height
    page_index = {v: i for i, v in enumerate(pages["version_id"].to_list())}

    foc = foc.filter(pl.col("version_id").is_in(list(page_index))).sort("focus_id")
    f_page = [page_index[v] for v in foc["version_id"].to_list()]
    f_text = [focus_text(r["summary"], r["tags"]) for r in foc.iter_rows(named=True)]
    f_summary = foc["summary"].to_list()
    f_vec = np.array(foc["vector"].to_list(), dtype=np.float32)

    # Position: the mean of a page's focus vectors, so every focus counts.
    page_vecs = np.zeros((n_pages, f_vec.shape[1]), dtype=np.float32)
    for k, i in enumerate(f_page):
        page_vecs[i] += f_vec[k]
    page_vecs /= np.clip(np.linalg.norm(page_vecs, axis=1, keepdims=True), 1e-9, None)
    coords = layout_2d(page_vecs)

    # Topics
    topic_of = merge_similar(f_vec, cluster_ids(f_vec, min_cluster_size))
    for t in set(topic_of.tolist()):
        if t >= 0 and len({f_page[k] for k in np.flatnonzero(topic_of == t)}) < cfg.min_topic_pages:
            topic_of[topic_of == t] = -1
    names = label_topics(f_text, f_page, topic_of)
    if llm_names and names and settings.llm_api_key:
        try:
            samples: dict[int, list[str]] = {}
            for t in names:
                idx = np.flatnonzero(topic_of == t)
                centroid = f_vec[idx].mean(axis=0)
                order = idx[np.argsort(-(f_vec[idx] @ centroid))][: cfg.topic_sample_size]
                samples[t] = [f_summary[k] for k in order]
            names |= name_topics(settings, samples)
        except Exception as e:  # fall back to keyword labels
            log.warning("LLM topic naming failed (%s); using keyword labels", type(e).__name__)
    centroids = {t: f_vec[topic_of == t].mean(axis=0) for t in names}

    weights = np.zeros((n_pages, max(names, default=-1) + 1))
    counts = np.bincount(f_page, minlength=n_pages)
    snippets: dict[tuple[int, int], str] = {}
    best: dict[tuple[int, int], float] = {}
    for k, (i, t) in enumerate(zip(f_page, topic_of.tolist(), strict=True)):
        if t < 0:
            continue
        weights[i, t] += 1 / counts[i]
        c = centroids[t]
        sim = float(f_vec[k] @ c / max(float(np.linalg.norm(c)), 1e-9))
        if sim > best.get((i, t), -1):
            best[(i, t)] = sim
            snippets[(i, t)] = f_summary[k]
    primary = (
        np.where(weights.sum(axis=1) > 0, weights.argmax(axis=1), -1)
        if names
        else np.full(n_pages, -1)
    )

    run_id = new_run_id()
    layouts = pl.DataFrame(
        {
            "run_id": [run_id] * n_pages,
            "id": pages["version_id"].to_list(),
            "x": coords[:, 0].astype(float).tolist(),
            "y": coords[:, 1].astype(float).tolist(),
            "cluster_id": [int(c) for c in primary],
            "cluster_label": [names.get(int(c)) for c in primary],
        },
        schema=LAYOUTS,
    )
    dataset.write_table("layouts", layouts)

    points = [
        {
            "id": r["version_id"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "name": r["name"],
            "city": r["city"],
            "country": r["country"],
            "url": r["url"],
            "captured_at": r["last_seen"],
            "excerpt": quotes.get(r["version_id"]) or _excerpt(r["text_md"]),
            "topics": sorted(
                (
                    {"id": t, "weight": round(float(weights[i, t]), 3), "snippet": snippets[(i, t)]}
                    for t in names
                    if weights[i, t] > 0
                ),
                key=lambda d: -float(d["weight"]),  # pyright: ignore[reportArgumentType]
            ),
        }
        for i, r in enumerate(pages.iter_rows(named=True))
    ]
    export = {
        "run_id": run_id,
        "topics": [
            {"id": t, "label": names[t], "pages": int((weights[:, t] > 0).sum())}
            for t in sorted(names)
        ],
        "points": points,
    }
    report = ClusterReport(run_id, n_pages, len(names), int((weights.sum(axis=1) == 0).sum()))
    return report, export


def write_export(export: dict[str, Any], path: str) -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export, ensure_ascii=False))
