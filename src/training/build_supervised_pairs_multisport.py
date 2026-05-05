"""
build_supervised_pairs_multisport.py
------------------------------------
Combine prebuilt supervised pair CSVs into one multisport pair CSV.

This does not build raw pairs. It only concatenates existing files that already
contain input_text, target_text, and event_type columns.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def combine_pair_csvs(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> None:
    frames = []

    for path in input_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pair CSV not found: {path}")

        df = pd.read_csv(path)

        required = {"input_text", "target_text", "event_type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if shuffle:
        combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Wrote {len(combined)} combined pairs to {output_path}")

    if "sport" in combined.columns:
        print("\nSport counts:")
        print(combined["sport"].fillna("cricket").value_counts().to_string())

    print("\nEvent counts:")
    print(combined["event_type"].value_counts().sort_index().to_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine cricket and soccer supervised pair CSVs."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more prebuilt pair CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output combined pair CSV path.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    combine_pair_csvs(
        args.input,
        args.output,
        shuffle=args.shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()