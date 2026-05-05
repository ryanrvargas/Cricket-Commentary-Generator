"""
build_supervised_pairs_controlled.py
------------------------------------
Build controlled prompt-target pairs for T5 fine-tuning.

This builder keeps the same prompt schema as build_supervised_pairs.py, but it
replaces raw commentary prose with one canonical event-faithful target per event.

Goal:
    Make the model learn the correct relationship between event_type/runs/wicket
    and the generated commentary line.

Example input_text:
    sport: cricket
    event_type: boundary_four
    batter: Mohammad Rizwan
    bowler: Mohammad Ali
    runs: 4
    wicket: none
    generate commentary:

Example target_text:
    Four. The ball reaches the boundary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.cricket.load_commentary import load_commentary_rows
from src.training.build_supervised_pairs import prompt_from_commentary_row


CANONICAL_TARGETS = {
    "dot_ball": "Dot ball. No run.",
    "single": "One run. The batters take a single.",
    "double": "Two runs. The batters come back for two.",
    "triple": "Three runs. The batters run hard.",
    "boundary_four": "Four. The ball reaches the fence.",
    "boundary_six": "Six. The ball goes into the stands.",
    "wicket_bowled": "Bowled. The stumps are hit.",
    "wicket_caught": "Caught. The catch is taken.",
    "wicket_lbw": "LBW. The batter is trapped in front.",
    "run_out": "Run out. The batter is short of the crease.",
    "wide": "Wide. Extra run conceded.",
    "no_ball": "No-ball. Extra run conceded.",
    "bye_or_legbye": "Extras. Runs are added without the bat.",
    "other": "Play continues.",
}


def controlled_target_from_row(row: dict[str, Any]) -> str:
    """
    Convert one parsed commentary row into a single canonical target sentence.

    This intentionally avoids random templates and player names because the
    earlier T5 runs were mixing suffixes across event classes.
    """
    event_type = row.get("event_type", "other")
    return CANONICAL_TARGETS.get(event_type, CANONICAL_TARGETS["other"])


def build_pairs_from_csv(
    csv_path: str | Path,
    *,
    target_mode: str = "controlled",
    min_target_words: int = 0,
    max_target_words: int = 0,
) -> list[dict[str, Any]]:
    """
    Build controlled prompt-target pairs from a commentary CSV file.

    The extra arguments are kept for compatibility with finetune_t5.py.
    """
    if target_mode != "controlled":
        raise ValueError("Controlled builder only supports target_mode='controlled'.")

    rows = load_commentary_rows(csv_path)
    pairs: list[dict[str, Any]] = []

    for row in rows:
        event_type = row.get("event_type", "other")
        target = controlled_target_from_row(row)

        if not target:
            continue

        pairs.append(
            {
                "input_text": prompt_from_commentary_row(row),
                "target_text": target,
                "event_type": event_type,
                "batter": row.get("batsman_name", ""),
                "bowler": row.get("bowler_name", ""),
                "play_type": row.get("play_type", ""),
                "target_mode": "canonical_controlled",
            }
        )

    return pairs


def write_pairs_csv(pairs: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Save pairs as CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(output_path, index=False)


def write_pairs_jsonl(pairs: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Save pairs as JSONL.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical controlled supervised pairs for T5 cricket fine-tuning."
    )
    parser.add_argument("--input", default="raw/train.csv")
    parser.add_argument("--output", default="raw/train_pairs_controlled.csv")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pairs = build_pairs_from_csv(args.input)

    if args.shuffle:
        random.Random(args.seed).shuffle(pairs)

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if args.format == "jsonl":
        write_pairs_jsonl(pairs, args.output)
    else:
        write_pairs_csv(pairs, args.output)

    print(f"Wrote {len(pairs)} canonical controlled supervised pairs to {args.output}")

    if pairs:
        print("\nExample input_text:\n")
        print(pairs[0]["input_text"])
        print("\nExample target_text:\n")
        print(pairs[0]["target_text"])


if __name__ == "__main__":
    main()