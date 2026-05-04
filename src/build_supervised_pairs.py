"""
build_supervised_pairs.py
-------------------------
Build supervised prompt-target pairs for fine-tuning a small text-to-text model.

This version intentionally uses a simpler prompt than the first experimental
version. The commentary CSV does not reliably contain live-match fields such as
over number or score context, so those fields are not included in the model
prompt. Keeping the training prompt and inference prompt aligned should make the
model easier to fine-tune and less likely to ignore the event type.

Example input_text:
    sport: cricket
    event_type: boundary_four
    batter: Mohammad Rizwan
    bowler: Mohammad Ali
    runs: 4
    wicket: none
    generate commentary:

Example target_text:
    Mohammad Rizwan leans into the drive and finds the boundary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Let this script work from either the project root or inside src/.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from load_commentary import load_commentary_rows  # noqa: E402

EXTRA_EVENT_TYPES = {"wide", "no_ball", "bye_or_legbye"}


def _clean_field(value: Any, default: str = "unknown") -> str:
    """
    Convert a field into a safe one-line string for a prompt.
    """
    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    return " ".join(text.split())


def _clean_target(value: Any) -> str:
    """
    Clean a target commentary sentence.
    """
    return _clean_field(value, default="")


def _infer_runs(row: dict[str, Any]) -> int:
    """
    Infer the most useful runs value for the simplified prompt.

    For extras, the commentary CSV total is not batter runs, so use 0. For normal
    scoring shots, use the parsed total_runs value.
    """
    total_runs = int(row.get("total_runs", 0) or 0)
    event_type = row.get("event_type", "other")

    if event_type in EXTRA_EVENT_TYPES:
        return 0

    return total_runs


def _wicket_text(row: dict[str, Any]) -> str:
    """
    Return a compact wicket description for the prompt.
    """
    dismissal_type = _clean_field(row.get("dismissal_type"), default="none")
    dismissal_flag = bool(row.get("dismissal_flag", False))

    if dismissal_flag and dismissal_type != "none":
        return dismissal_type

    return "none"


def prompt_from_fields(
    *,
    sport: str = "cricket",
    event_type: str,
    batter: str,
    bowler: str,
    runs: int | str = 0,
    wicket: str = "none",
) -> str:
    """
    Build the text-to-text input prompt used for fine-tuning and inference.

    Keeping this format in one function matters. The model should see the same
    input shape during training that it sees during inference.
    """
    return "\n".join(
        [
            f"sport: {_clean_field(sport, default='cricket')}",
            f"event_type: {_clean_field(event_type, default='other')}",
            f"batter: {_clean_field(batter)}",
            f"bowler: {_clean_field(bowler)}",
            f"runs: {_clean_field(runs, default='0')}",
            f"wicket: {_clean_field(wicket, default='none')}",
            "generate commentary:",
        ]
    )


def prompt_from_commentary_row(row: dict[str, Any]) -> str:
    """
    Convert one parsed commentary CSV row into a model input prompt.
    """
    return prompt_from_fields(
        event_type=row.get("event_type", "other"),
        batter=row.get("batsman_name", "unknown"),
        bowler=row.get("bowler_name", "unknown"),
        runs=_infer_runs(row),
        wicket=_wicket_text(row),
    )


def prompt_from_match_event(
    event: dict[str, Any],
    event_type: str,
    context: dict[str, Any] | None = None,
) -> str:
    """
    Convert one live/simulated Cricsheet event dictionary into a model prompt.

    The context parameter is accepted for compatibility with model_infer.py, but
    this simplified prompt intentionally does not include score or over fields.
    """
    wicket = event.get("wicket_type") or "none"

    return prompt_from_fields(
        event_type=event_type,
        batter=event.get("batter", "unknown"),
        bowler=event.get("bowler", "unknown"),
        runs=event.get("runs_off_bat", 0),
        wicket=wicket,
    )


def build_pairs_from_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Build prompt-target pairs from a commentary CSV file.

    Returns dictionaries with input_text, target_text, and a few metadata fields
    that are useful for debugging and evaluation.
    """
    rows = load_commentary_rows(csv_path)
    pairs: list[dict[str, Any]] = []

    for row in rows:
        target = _clean_target(row.get("commentary"))
        if not target:
            continue

        pairs.append(
            {
                "input_text": prompt_from_commentary_row(row),
                "target_text": target,
                "event_type": row.get("event_type", "other"),
                "batter": row.get("batsman_name", ""),
                "bowler": row.get("bowler_name", ""),
                "play_type": row.get("play_type", ""),
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
        description="Build supervised input-output pairs for T5 fine-tuning."
    )
    parser.add_argument(
        "--input",
        default="raw/train.csv",
        help="Path to a commentary CSV file such as raw/train.csv.",
    )
    parser.add_argument(
        "--output",
        default="raw/train_pairs.csv",
        help="Where to save the generated pairs.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output file format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of pairs to write. Use 0 for all pairs.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle pairs before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set.",
    )
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

    print(f"Wrote {len(pairs)} supervised pairs to {args.output}")
    if pairs:
        print("\nExample input_text:\n")
        print(pairs[0]["input_text"])
        print("\nExample target_text:\n")
        print(pairs[0]["target_text"])


if __name__ == "__main__":
    main()
