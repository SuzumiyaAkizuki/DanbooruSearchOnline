"""Validated loader for the local Danbooru Tag Alias snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ALIAS_SCHEMA_VERSION = 1
ALIAS_COLUMNS = [
    "id",
    "antecedent_name",
    "consequent_name",
    "status",
    "updated_at",
    "target_in_tag_db",
]


class TagAliasSnapshotError(ValueError):
    """Raised when the Parquet and metadata do not form a valid snapshot."""


@dataclass(frozen=True)
class TagAliasRecord:
    consequent_name: str
    target_in_tag_db: bool


@dataclass(frozen=True)
class TagAliasSnapshot:
    alias_by_name: dict[str, TagAliasRecord]
    metadata: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.alias_by_name)


def load_tag_alias_snapshot(
    parquet_path: str | Path,
    metadata_path: str | Path,
) -> TagAliasSnapshot:
    parquet_path = Path(parquet_path)
    metadata_path = Path(metadata_path)
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise TagAliasSnapshotError("Tag Alias snapshot files are missing")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TagAliasSnapshotError("Tag Alias metadata is invalid") from exc
    if not isinstance(metadata, dict):
        raise TagAliasSnapshotError("Tag Alias metadata must be an object")
    if metadata.get("schema_version") != ALIAS_SCHEMA_VERSION:
        raise TagAliasSnapshotError("Unsupported Tag Alias schema_version")

    try:
        aliases = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise TagAliasSnapshotError("Unable to read Tag Alias Parquet") from exc
    if list(aliases.columns) != ALIAS_COLUMNS:
        raise TagAliasSnapshotError("Unexpected Tag Alias columns")
    if not pd.api.types.is_integer_dtype(aliases["id"]):
        raise TagAliasSnapshotError("Tag Alias id must be an integer column")
    for column in ("antecedent_name", "consequent_name", "status", "updated_at"):
        if not pd.api.types.is_string_dtype(aliases[column]):
            raise TagAliasSnapshotError(f"Tag Alias {column} must be a string column")
    if not pd.api.types.is_bool_dtype(aliases["target_in_tag_db"]):
        raise TagAliasSnapshotError("Tag Alias target_in_tag_db must be boolean")
    if len(aliases) != int(metadata.get("active_count", -1)):
        raise TagAliasSnapshotError("Tag Alias row count does not match metadata")
    if aliases[ALIAS_COLUMNS].isna().any().any():
        raise TagAliasSnapshotError("Tag Alias snapshot contains null fields")
    if not aliases["status"].eq("active").all():
        raise TagAliasSnapshotError("Tag Alias snapshot contains inactive rows")
    if not aliases["antecedent_name"].is_unique:
        raise TagAliasSnapshotError("Tag Alias antecedent_name is not unique")
    if aliases["antecedent_name"].eq(aliases["consequent_name"]).any():
        raise TagAliasSnapshotError("Tag Alias snapshot contains self mappings")

    target_count = int(aliases["target_in_tag_db"].sum())
    missing_count = len(aliases) - target_count
    if target_count != int(metadata.get("target_in_tag_db_count", -1)):
        raise TagAliasSnapshotError("Tag Alias target count does not match metadata")
    if missing_count != int(metadata.get("target_missing_count", -1)):
        raise TagAliasSnapshotError("Tag Alias missing-target count does not match metadata")
    if len(aliases) != target_count + missing_count:
        raise TagAliasSnapshotError("Tag Alias target counts are inconsistent")

    antecedents = set(aliases["antecedent_name"].astype(str))
    chained_targets = antecedents.intersection(
        set(aliases["consequent_name"].astype(str))
    )
    if chained_targets:
        raise TagAliasSnapshotError("Tag Alias snapshot contains active chains")

    alias_by_name = {
        str(row.antecedent_name): TagAliasRecord(
            consequent_name=str(row.consequent_name),
            target_in_tag_db=bool(row.target_in_tag_db),
        )
        for row in aliases.itertuples(index=False)
    }
    return TagAliasSnapshot(alias_by_name=alias_by_name, metadata=metadata)
