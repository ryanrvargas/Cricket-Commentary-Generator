"""
build_supervised_pairs.py
-------------------------
Build supervised prompt-target pairs for fine-tuning a small text-to-text model.

This version uses a simplified prompt and adds light factual prefixes to targets.
The goal is to teach the model that event labels must be reflected in the output.
For example, wicket_bowled targets should explicitly include a bowled/stumps idea,
and boundary_four targets should explicitly include a four/boundary idea.

Example input_text:
    sport: cricket
    event_type: wicket_bowled
    batter: UT Khawaja
    bowler: GJ Maxwell
    runs: 0
    wicket: bowled
    generate commentary:

Example target_text:
    Bowled. UT Khawaja is beaten and the stumps are hit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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

from cricket.load_commentary import load_commentary_rows  # noqa: E402

EXTRA_EVENT_TYPES = {"wide", "no_ball", "bye_or_legbye"}

EVENT_KEYWORDS = {
    "boundary_four": ["four", "boundary", "rope", "fence"],
    "boundary_six": ["six", "maximum", "stands", "crowd", "rope"],
    "wicket_bowled": ["bowled", "stumps", "cleaned", "knocked", "castle"],
    "wicket_caught": ["caught", "catch", "taken", "edge", "nick", "fielder"],
    "wicket_lbw": ["lbw", "trapped", "plumb", "front"],
    "run_out": ["run out", "short", "direct hit"],
    "wide": ["wide"],
    "no_ball": ["no-ball", "no ball", "overstepped", "overstep"],
    "bye_or_legbye": ["bye", "byes", "leg bye", "legbyes", "extras", "pads"],
    "single": ["single", "one"],
    "double": ["two", "couple"],
    "triple": ["three"],
}

EVENT_PREFIX = {
    "boundary_four": "Four. ",
    "boundary_six": "Six. ",
    "wicket_bowled": "Bowled. ",
    "wicket_caught": "Caught. ",
    "wicket_lbw": "LBW. ",
    "run_out": "Run out. ",
    "wide": "Wide. ",
    "no_ball": "No-ball. ",
    "bye_or_legbye": "Extras. ",
    "single": "One run. ",
    "double": "Two runs. ",
    "triple": "Three runs. ",
}


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
    text = _clean_field(value, default="")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalized(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_event_keyword(event_type: str, target: str) -> bool:
    """
    Return True if a target already mentions the key factual idea.
    """
    keywords = EVENT_KEYWORDS.get(event_type, [])
    normalized = _normalized(target)
    return any(keyword in normalized for keyword in keywords)


def _add_fact_prefix(event_type: str, target: str) -> tuple[str, bool]:
    """
    Add a short factual prefix when the target does not explicitly reflect the event.

    This is intentional. The original commentary text can be stylish but vague.
    For fine-tuning, we need the target to teach the model that a wicket_bowled
    prompt should mention a bowled/stumps idea and a boundary_four prompt should
    mention a four/boundary idea.
    """
    if not target:
        return target, False

    if _has_event_keyword(event_type, target):
        return target, False

    prefix = EVENT_PREFIX.get(event_type)
    if not prefix:
        return target, False

    return prefix + target, True


def _target_word_count(target: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", target))


def _is_reasonable_target(target: str, min_words: int, max_words: int) -> bool:
    """
    Filter out very weak targets before training.
    """
    if not target:
        return False

    word_count = _target_word_count(target)
    if word_count < min_words:
        return False
    if max_words > 0 and word_count > max_words:
        return False

    # Drop obvious junk-like rows.
    letters = re.findall(r"[A-Za-z]", target)
    if len(letters) < 8:
        return False

    return True


def _infer_runs(row: dict[str, Any]) -> int:
    """
    Infer the most useful runs value for the simplified prompt.
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


def build_pairs_from_csv(
    csv_path: str | Path,
    *,
    target_mode: str = "fact_prefix",
    min_target_words: int = 4,
    max_target_words: int = 40,
) -> list[dict[str, Any]]:
    """
    Build prompt-target pairs from a commentary CSV file.

    target_mode:
        original: use cleaned commentary as-is.
        fact_prefix: add short factual prefixes when targets do not explicitly
                     mention the event fact.
    """
    rows = load_commentary_rows(csv_path)
    pairs: list[dict[str, Any]] = []

    for row in rows:
        event_type = row.get("event_type", "other")
        target = _clean_target(row.get("commentary"))

        if not _is_reasonable_target(target, min_target_words, max_target_words):
            continue

        was_augmented = False
        if target_mode == "fact_prefix":
            target, was_augmented = _add_fact_prefix(event_type, target)
        elif target_mode != "original":
            raise ValueError("target_mode must be 'original' or 'fact_prefix'.")

        pairs.append(
            {
                "input_text": prompt_from_commentary_row(row),
                "target_text": target,
                "event_type": event_type,
                "batter": row.get("batsman_name", ""),
                "bowler": row.get("bowler_name", ""),
                "play_type": row.get("play_type", ""),
                "target_was_augmented": was_augmented,
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
    parser.add_argument("--input", default="raw/train.csv")
    parser.add_argument("--output", default="raw/train_pairs.csv")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=["original", "fact_prefix"], default="fact_prefix")
    parser.add_argument("--min-target-words", type=int, default=4)
    parser.add_argument("--max-target-words", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pairs = build_pairs_from_csv(
        args.input,
        target_mode=args.target_mode,
        min_target_words=args.min_target_words,
        max_target_words=args.max_target_words,
    )

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
        augmented = sum(1 for pair in pairs if pair.get("target_was_augmented"))
        print(f"\nTargets with factual prefixes added: {augmented}")


if __name__ == "__main__":
    main()
