"""
build_supervised_pairs_controlled.py
------------------------------------
Build controlled prompt-target pairs for T5 fine-tuning.

This builder keeps the same prompt schema as build_supervised_pairs.py, but it
replaces raw commentary prose with short controlled event-faithful targets.

Goal:
    Make the model learn the correct relationship between event_type/runs/wicket
    and the generated commentary line.

Example:
    event_type: boundary_four -> "Four. Rizwan finds the boundary."
    event_type: dot_ball      -> "Dot ball. Rizwan defends."
    event_type: wicket_bowled -> "Bowled. Khawaja is beaten and the stumps are hit."
"""

from __future__ import annotations

import argparse
import hashlib
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


def _clean_name(value: Any, default: str = "The batter") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return " ".join(text.split())


def _stable_choice(options: list[str], *keys: Any) -> str:
    """
    Pick a deterministic template so the same row always gets the same target.

    This gives variety without making the dataset randomly change every run.
    """
    if not options:
        return ""

    key_text = "|".join(str(key) for key in keys)
    digest = hashlib.md5(key_text.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(options)
    return options[index]


def controlled_target_from_row(row: dict[str, Any]) -> str:
    """
    Convert one parsed commentary row into a controlled target sentence.

    The target must be faithful to the event label. Do not use rich shot details
    that are not present in the prompt.
    """
    event_type = row.get("event_type", "other")
    batter = _clean_name(row.get("batsman_name"))
    bowler = _clean_name(row.get("bowler_name"), default="the bowler")

    templates = {
        "dot_ball": [
            f"Dot ball. {batter} defends.",
            f"Dot ball. {batter} plays with no run.",
            f"No run. {batter} cannot score.",
        ],
        "single": [
            f"One run. {batter} picks up a single.",
            f"One run. {batter} works it away.",
            f"One run. {batter} rotates the strike.",
        ],
        "double": [
            f"Two runs. {batter} comes back for two.",
            f"Two runs. {batter} places it into the gap.",
            f"Two runs. {batter} runs well.",
        ],
        "triple": [
            f"Three runs. {batter} runs hard.",
            f"Three runs. {batter} finds the gap.",
            f"Three runs. {batter} comes back for three.",
        ],
        "boundary_four": [
            f"Four. {batter} finds the boundary.",
            f"Four. {batter} drives it away.",
            f"Four. {batter} gets it to the rope.",
        ],
        "boundary_six": [
            f"Six. {batter} clears the rope.",
            f"Six. {batter} sends it over the boundary.",
            f"Six. {batter} launches it away.",
        ],
        "wicket_bowled": [
            f"Bowled. The stumps are hit.",
            f"Bowled. {bowler} hits the stumps.",
            f"Bowled. {batter} is beaten.",
        ],
        "wicket_caught": [
            f"Caught. The catch is taken.",
            f"Caught. {batter} is caught off {bowler}.",
            f"Caught. {bowler} gets the wicket.",
        ],
        "wicket_lbw": [
            f"LBW. {batter} is trapped in front.",
            f"LBW. The umpire gives it.",
            f"LBW. {bowler} strikes.",
        ],
        "run_out": [
            f"Run out. The batter is short of the crease.",
            f"Run out. Sharp fielding gets the wicket.",
            f"Run out. The throw beats the batter.",
        ],
        "wide": [
            f"Wide. {bowler} misses the line.",
            f"Wide. Extra run conceded.",
            f"Wide. That is too far away.",
        ],
        "no_ball": [
            f"No-ball. {bowler} oversteps.",
            f"No-ball. Extra run conceded.",
            f"No-ball. That must be bowled again.",
        ],
        "bye_or_legbye": [
            "Extras. They pick up runs not off the bat.",
            "Extras. The batters take the extra run.",
            "Extras. Runs are added without a shot.",
        ],
        "other": [
            "The ball is played.",
            f"{batter} plays the delivery.",
            f"{bowler} completes the delivery.",
        ],
    }

    options = templates.get(event_type, templates["other"])
    return _stable_choice(
        options,
        event_type,
        row.get("batsman_name", ""),
        row.get("bowler_name", ""),
        row.get("play_type", ""),
        row.get("total_runs", ""),
        row.get("dismissal_type", ""),
    )


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
                "target_mode": "controlled",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build controlled supervised pairs for T5 cricket fine-tuning."
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

    print(f"Wrote {len(pairs)} controlled supervised pairs to {args.output}")

    if pairs:
        print("\nExample input_text:\n")
        print(pairs[0]["input_text"])
        print("\nExample target_text:\n")
        print(pairs[0]["target_text"])


if __name__ == "__main__":
    main()