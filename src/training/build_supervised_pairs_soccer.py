"""
build_supervised_pairs_soccer.py
--------------------------------
Build controlled soccer prompt-target pairs for T5 fine-tuning.

This is the soccer equivalent of build_supervised_pairs_controlled.py.

It reads the YallaShoot-style soccer commentary CSV and creates simple
event-faithful training pairs.

Example input_text:
    sport: soccer
    event_type: goal
    player: Lionel Messi
    team: Barcelona
    minute: 73
    generate commentary:

Example target_text:
    Goal for Barcelona. Lionel Messi finds the net.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.soccer.load_soccer_commentary import load_soccer_commentary_rows


def _clean_field(value: Any, default: str = "unknown") -> str:
    """
    Convert a CSV value into a safe one-line prompt field.
    Handles blanks and pandas NaN values.
    """
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return default

    return " ".join(text.split())


def prompt_from_soccer_row(row: dict[str, Any]) -> str:
    """
    Build a soccer text-to-text prompt.

    This intentionally uses soccer-shaped fields instead of cricket fields like
    batter, bowler, runs, and wicket.
    """
    event_type = _clean_field(row.get("event_type"), default="other")
    player = _clean_field(row.get("player"), default="The player")
    team = _clean_field(row.get("team"), default="the team")
    minute = _clean_field(row.get("minute"), default="0")

    return "\n".join(
        [
            "sport: soccer",
            f"event_type: {event_type}",
            f"player: {player}",
            f"team: {team}",
            f"minute: {minute}",
            "generate commentary:",
        ]
    )


def controlled_soccer_target_from_row(row: dict[str, Any]) -> str:
    """
    Convert one soccer commentary row into a controlled target sentence.
    """
    event_type = _clean_field(row.get("event_type"), default="other")
    player = _clean_field(row.get("player"), default="The player")
    team = _clean_field(row.get("team"), default="the team")

    if event_type == "goal":
        return f"Goal for {team}. {player} finds the net."

    if event_type == "shot":
        return f"Shot. {player} gets an effort away for {team}."

    if event_type == "save":
        return f"Save. {player} keeps it out."

    if event_type == "pass":
        return f"Pass. {player} keeps play moving for {team}."

    if event_type == "corner":
        return f"Corner for {team}."

    if event_type == "free_kick":
        return f"Free kick for {team}."

    if event_type == "foul":
        return f"Foul by {player}."

    if event_type == "yellow_card":
        return f"Yellow card shown to {player}."

    if event_type == "red_card":
        return f"Red card shown to {player}."

    if event_type == "offside":
        return f"Offside against {team}."

    if event_type == "substitution":
        return f"Substitution for {team}."

    return f"Play continues for {team}."


def build_pairs_from_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Build soccer prompt-target pairs from a YallaShoot-style commentary CSV.
    """
    rows = load_soccer_commentary_rows(csv_path)
    pairs: list[dict[str, Any]] = []

    for row in rows:
        event_type = _clean_field(row.get("event_type"), default="other")
        target = controlled_soccer_target_from_row(row)

        if not target:
            continue

        pairs.append(
            {
                "input_text": prompt_from_soccer_row(row),
                "target_text": target,
                "sport": "soccer",
                "event_type": event_type,
                "player": _clean_field(row.get("player"), default=""),
                "team": _clean_field(row.get("team"), default=""),
                "minute": _clean_field(row.get("minute"), default="0"),
                "target_mode": "soccer_controlled",
            }
        )

    return pairs


def write_pairs_csv(pairs: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(output_path, index=False)


def write_pairs_jsonl(pairs: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _print_event_counts(pairs: list[dict[str, Any]]) -> None:
    counts = Counter(pair.get("event_type", "other") for pair in pairs)

    print("\nSoccer pair event counts:")
    for event_type, count in sorted(counts.items()):
        print(f"  {event_type}: {count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build controlled supervised pairs for soccer T5 fine-tuning."
    )
    parser.add_argument(
        "--input",
        default="raw_soccer/yallashoot/commentary_train_augmented.csv",
        help="Path to the YallaShoot soccer commentary CSV.",
    )
    parser.add_argument(
        "--output",
        default="raw_soccer/yallashoot/soccer_pairs_controlled.csv",
        help="Output path for soccer controlled pairs.",
    )
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-event-counts", action="store_true")
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

    print(f"Wrote {len(pairs)} soccer controlled supervised pairs to {args.output}")

    if args.show_event_counts:
        _print_event_counts(pairs)

    if pairs:
        print("\nExample input_text:\n")
        print(pairs[0]["input_text"])
        print("\nExample target_text:\n")
        print(pairs[0]["target_text"])


if __name__ == "__main__":
    main()