"""Compact, human-readable formatting for result tables and metadata.

Raw predictions remain float32 in compressed NPZ files so evaluation can be
recomputed without quantisation.  Only text artifacts are rounded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULT_DECIMALS = 3
CSV_FLOAT_FORMAT = f"%.{RESULT_DECIMALS}f"


def round_result_numbers(value: Any, decimals: int = RESULT_DECIMALS) -> Any:
    """Recursively round floating-point values while preserving container shape."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return round(number, decimals) if math.isfinite(number) else number
    if isinstance(value, dict):
        return {key: round_result_numbers(item, decimals) for key, item in value.items()}
    if isinstance(value, list):
        return [round_result_numbers(item, decimals) for item in value]
    if isinstance(value, tuple):
        return tuple(round_result_numbers(item, decimals) for item in value)
    return value


def save_result_json(path: Path, payload: Any) -> None:
    """Write JSON metadata with at most three decimal places for floats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(round_result_numbers(payload), indent=2),
        encoding="utf-8",
    )


def save_result_table(table: pd.DataFrame, path: Path) -> None:
    """Write a result CSV using three decimal places for floating columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)


def round_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Round one streaming CSV row without changing integer/string fields."""
    return {key: round_result_numbers(value) for key, value in row.items()}
