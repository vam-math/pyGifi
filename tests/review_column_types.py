"""
Interactive helper to infer and optionally correct column roles for a CSV file.

Usage:
    python tests/review_column_types.py path/to/data.csv
    python tests/review_column_types.py path/to/data.csv --output column_kinds.json

Press Enter at any prompt to accept the inferred role.
If stdin is non-interactive, the script proceeds with the inferred roles.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pygifi.utils.type_inference import prepare_dataframe_with_inference


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python tests/review_column_types.py <csv-path> [--output path.json]")
        return 1

    csv_path = argv[1]
    output_path = None
    if "--output" in argv:
        flag_index = argv.index("--output")
        if flag_index + 1 >= len(argv):
            print("Missing path after --output")
            return 1
        output_path = argv[flag_index + 1]

    df = pd.read_csv(csv_path)
    interactive = bool(sys.stdin.isatty())
    _, resolved, inferred = prepare_dataframe_with_inference(
        df,
        interactive=interactive,
    )

    print("\nFinal column roles:")
    for col in df.columns:
        info = inferred[str(col)]
        print(
            f"  {col}: {resolved[str(col)]} "
            f"(inferred={info.kind}, unique={info.unique_count}, non-null={info.non_null_count})"
        )

    if output_path:
        payload: Dict[str, Dict[str, object]] = {
            str(col): {
                "kind": resolved[str(col)],
                "inferred_kind": inferred[str(col)].kind,
                "reasons": list(inferred[str(col)].reasons),
                "unique_count": inferred[str(col)].unique_count,
                "non_null_count": inferred[str(col)].non_null_count,
            }
            for col in df.columns
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved column-role report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
