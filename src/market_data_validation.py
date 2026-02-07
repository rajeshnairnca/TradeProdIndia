from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_INDEX_NAMES = ("date", "ticker")


def validate_market_data_frame(
    df: pd.DataFrame,
    *,
    source: str | Path = "market data",
    required_columns: list[str] | tuple[str, ...] | None = None,
    require_non_empty: bool = True,
) -> None:
    """Fail fast when the parquet shape/index is not what pipelines expect."""
    source_label = str(source)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{source_label}: expected pandas DataFrame, got {type(df).__name__}.")
    if require_non_empty and df.empty:
        raise ValueError(f"{source_label}: dataset is empty.")
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"{source_label}: expected MultiIndex {EXPECTED_INDEX_NAMES}, got {type(df.index).__name__}."
        )
    index_names = tuple(df.index.names)
    if index_names != EXPECTED_INDEX_NAMES:
        raise ValueError(
            f"{source_label}: expected index names {EXPECTED_INDEX_NAMES}, got {index_names}."
        )
    if df.index.has_duplicates:
        dup_count = int(df.index.duplicated().sum())
        raise ValueError(f"{source_label}: found {dup_count} duplicated [date, ticker] rows.")

    try:
        pd.to_datetime(df.index.get_level_values("date"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{source_label}: date index level is not datetime-compatible.") from exc

    if required_columns:
        missing_cols = sorted(set(required_columns) - set(df.columns))
        if missing_cols:
            raise ValueError(
                f"{source_label}: missing required columns: {', '.join(missing_cols)}."
            )
