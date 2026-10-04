"""Storage adapters: fsspec for objects and Parquet tables.

``Storage`` wraps one fsspec location. The pipeline uses two: a dataset (tables) and a bucket
(raw HTML). Both may be local directories (development, tests) or Hugging Face / S3 URLs.
"""

import io
from typing import Any

import fsspec
import polars as pl
from fsspec import AbstractFileSystem


class Storage:
    def __init__(self, url: str, **storage_options: Any) -> None:
        fs, root = fsspec.core.url_to_fs(url, **storage_options)  # pyright: ignore[reportUnknownMemberType]
        self.fs: AbstractFileSystem = fs
        self.root: str = str(root).rstrip("/")
        self.url = url

    def _path(self, key: str) -> str:
        return f"{self.root}/{key.lstrip('/')}"

    # -- objects ---------------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        return bool(self.fs.exists(self._path(key)))

    def read_bytes(self, key: str) -> bytes:
        data = self.fs.cat_file(self._path(key))
        return data.encode() if isinstance(data, str) else data

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        parent = path.rsplit("/", 1)[0]
        self.fs.makedirs(parent, exist_ok=True)
        self.fs.pipe_file(path, data)

    def delete(self, key: str) -> None:
        if self.exists(key):
            self.fs.rm_file(self._path(key))

    def list(self, prefix: str) -> list[str]:
        """Keys (relative to the root) of all files under ``prefix``, sorted."""
        base = self._path(prefix)
        if not self.fs.exists(base):
            return []
        found = [str(f) for f in self.fs.find(base)]
        # Some backends omit the leading slash/protocol; normalize relative to root.
        root = self.root.lstrip("/")
        return sorted(f.lstrip("/").removeprefix(root).lstrip("/") for f in found)

    # -- tables ----------------------------------------------------------------------------

    def write_table(self, name: str, df: pl.DataFrame) -> None:
        buf = io.BytesIO()
        df.write_parquet(buf)
        self.write_bytes(f"{name}.parquet", buf.getvalue())

    def read_table(self, name: str) -> pl.DataFrame | None:
        key = f"{name}.parquet"
        if not self.exists(key):
            return None
        return pl.read_parquet(io.BytesIO(self.read_bytes(key)))

    def read_tables(self, prefix: str) -> pl.DataFrame | None:
        """Concatenate every Parquet file under ``prefix`` (an append-only partitioned table)."""
        frames = [
            pl.read_parquet(io.BytesIO(self.read_bytes(key)))
            for key in self.list(prefix)
            if key.endswith(".parquet")
        ]
        return pl.concat(frames, how="diagonal_relaxed") if frames else None

    def write_json(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode())

    def read_text(self, key: str) -> str | None:
        return self.read_bytes(key).decode() if self.exists(key) else None
