"""
Utilities for inferring column roles in mixed-type datasets.

The heuristics here are intentionally conservative:
- obvious strings / booleans are treated as categorical
- high-cardinality real-valued data stays numerical
- low-cardinality integer-like data is treated as categorical

Callers can optionally let a human review the inferred roles before
coercing the DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


ColumnKind = str


@dataclass(frozen=True)
class ColumnInference:
    kind: ColumnKind
    reasons: Tuple[str, ...]
    unique_count: int
    non_null_count: int


def _is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series.dropna(), errors="coerce")
    if numeric.empty:
        return False
    return bool((numeric.round() == numeric).all())


def _looks_like_identifier(name: str) -> bool:
    lowered = name.strip().lower()
    hints = ("id", "code", "zip", "zipcode", "postal", "pin")
    return any(token in lowered for token in hints)


def infer_column_kind(
    series: pd.Series,
    *,
    categorical_unique_threshold: int = 12,
    categorical_ratio_threshold: float = 0.05,
) -> ColumnInference:
    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    non_null_count = int(non_null.shape[0])
    unique_ratio = (unique_count / non_null_count) if non_null_count else 0.0
    reasons: List[str] = []
    name = str(series.name) if series.name is not None else "<unnamed>"

    if non_null_count == 0:
        reasons.append("all values are missing; defaulting to categorical")
        return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)

    if isinstance(series.dtype, pd.CategoricalDtype):
        reasons.append("dtype is already categorical")
        return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)

    if is_bool_dtype(series.dtype):
        reasons.append("boolean columns are discrete categories")
        return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)

    if is_string_dtype(series.dtype) or is_object_dtype(series.dtype):
        parsed = pd.to_numeric(non_null, errors="coerce")
        if parsed.notna().all():
            reasons.append("string/object values are all parseable as numbers")
            if _is_integer_like(parsed):
                if unique_count <= categorical_unique_threshold or unique_ratio <= categorical_ratio_threshold:
                    reasons.append("integer-like values have low cardinality")
                    return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)
                reasons.append("integer-like values have high cardinality")
                return ColumnInference("numerical", tuple(reasons), unique_count, non_null_count)

            if unique_count <= categorical_unique_threshold and unique_ratio <= categorical_ratio_threshold:
                reasons.append("few distinct numeric-like values relative to rows")
                return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)

            reasons.append("continuous numeric-like values suggest a metric column")
            return ColumnInference("numerical", tuple(reasons), unique_count, non_null_count)

        reasons.append("contains non-numeric labels")
        return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)

    if is_numeric_dtype(series.dtype):
        if is_float_dtype(series.dtype):
            reasons.append("float dtype indicates metric data")
            if unique_count <= categorical_unique_threshold and unique_ratio <= categorical_ratio_threshold and _is_integer_like(non_null):
                reasons.append("values are integer-like with low cardinality")
                return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)
            return ColumnInference("numerical", tuple(reasons), unique_count, non_null_count)

        if is_integer_dtype(series.dtype):
            reasons.append("integer dtype can be either coded categories or metric counts")
            if _looks_like_identifier(name):
                reasons.append("column name looks like an identifier/code")
                return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)
            if unique_count <= categorical_unique_threshold:
                reasons.append("few distinct integer values")
                return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)
            if unique_ratio <= categorical_ratio_threshold:
                reasons.append("integer values repeat heavily across rows")
                return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)
            reasons.append("many distinct integer values suggest metric/count data")
            return ColumnInference("numerical", tuple(reasons), unique_count, non_null_count)

    reasons.append("fallback to categorical for unsupported dtype")
    return ColumnInference("categorical", tuple(reasons), unique_count, non_null_count)


def infer_column_kinds(
    df: pd.DataFrame,
    *,
    categorical_unique_threshold: int = 12,
    categorical_ratio_threshold: float = 0.05,
) -> Dict[str, ColumnInference]:
    return {
        str(col): infer_column_kind(
            df[col],
            categorical_unique_threshold=categorical_unique_threshold,
            categorical_ratio_threshold=categorical_ratio_threshold,
        )
        for col in df.columns
    }


def review_column_kinds(
    df: pd.DataFrame,
    inferred: Dict[str, ColumnInference],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Dict[str, ColumnKind]:
    """
    Let a human accept or override inferred roles.

    Pressing Enter accepts the inferred role.
    Entering `c`/`categorical` or `n`/`numerical` overrides it.
    EOF or non-interactive stdin is treated as "accept all inferred roles".
    """
    resolved: Dict[str, ColumnKind] = {}
    for col in df.columns:
        info = inferred[str(col)]
        reason_text = "; ".join(info.reasons)
        output_fn(
            f"[{col}] inferred={info.kind} "
            f"(unique={info.unique_count}, non-null={info.non_null_count})"
        )
        output_fn(f"  reasons: {reason_text}")
        try:
            answer = input_fn("  Press Enter to accept, or type categorical/numerical: ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("", "accept", "a"):
            resolved[str(col)] = info.kind
        elif answer in ("c", "cat", "categorical"):
            resolved[str(col)] = "categorical"
        elif answer in ("n", "num", "numerical", "numeric"):
            resolved[str(col)] = "numerical"
        else:
            output_fn("  Unrecognized input; keeping inferred role.")
            resolved[str(col)] = info.kind
    return resolved


def coerce_dataframe_by_kinds(
    df: pd.DataFrame,
    kinds: Dict[str, ColumnKind],
) -> pd.DataFrame:
    coerced = df.copy()
    for col in coerced.columns:
        role = kinds.get(str(col), "categorical")
        if role == "numerical":
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
        else:
            coerced[col] = coerced[col].astype("category")
    return coerced


def prepare_dataframe_with_inference(
    df: pd.DataFrame,
    *,
    interactive: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    categorical_unique_threshold: int = 12,
    categorical_ratio_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, ColumnKind], Dict[str, ColumnInference]]:
    inferred = infer_column_kinds(
        df,
        categorical_unique_threshold=categorical_unique_threshold,
        categorical_ratio_threshold=categorical_ratio_threshold,
    )
    if interactive:
        kinds = review_column_kinds(df, inferred, input_fn=input_fn, output_fn=output_fn)
    else:
        kinds = {name: info.kind for name, info in inferred.items()}
    coerced = coerce_dataframe_by_kinds(df, kinds)
    return coerced, kinds, inferred
