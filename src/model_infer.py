"""
model_infer.py
--------------
Generate commentary from a fine-tuned T5/FLAN-T5 checkpoint.

The main function accepts a Cricsheet-style event row, formats it with the same
simplified prompt structure used during fine-tuning, and asks the model to
generate one commentary line.

Example using a saved match file:
    python src/model_infer.py --checkpoint models/t5-cricket-commentary --match raw/1527575.json --event-index 2 --show-prompt

Example using manual fields:
    python src/model_infer.py --checkpoint models/t5-cricket-commentary --event-type boundary_four --batter "Mohammad Rizwan" --bowler "Mohammad Ali" --runs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Let this script work from either the project root or inside src/.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_supervised_pairs import prompt_from_match_event  # noqa: E402
from classify_event import classify_event  # noqa: E402
from load_events import load_match_events  # noqa: E402


def _require_transformers():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install them with:\n"
            "pip install torch transformers sentencepiece"
        ) from exc

    return AutoModelForSeq2SeqLM, AutoTokenizer


def build_context_for_event(events: list[dict[str, Any]], event_index: int) -> dict[str, Any]:
    """
    Build score context through the selected event index.

    The context is score-after-ball because that matches the existing demo style
    where commentary can say things like "That moves them to 4/0.".
    """
    if event_index < 0 or event_index >= len(events):
        raise IndexError(f"event_index {event_index} is outside 0..{len(events) - 1}")

    current_innings = None
    innings_score = 0
    innings_wickets = 0

    for index, event in enumerate(events):
        if current_innings != event.get("innings"):
            current_innings = event.get("innings")
            innings_score = 0
            innings_wickets = 0

        runs_on_ball = int(event.get("runs_off_bat", 0) or 0) + int(event.get("extras", 0) or 0)
        wickets_lost_on_ball = 1 if event.get("wicket_type") else 0

        innings_score += runs_on_ball
        innings_wickets += wickets_lost_on_ball

        if index == event_index:
            return {
                "innings": event.get("innings"),
                "over": event.get("over"),
                "ball_in_over": event.get("ball_in_over"),
                "runs_on_ball": runs_on_ball,
                "wickets_lost_on_ball": wickets_lost_on_ball,
                "innings_score": innings_score,
                "innings_wickets": innings_wickets,
            }

    raise RuntimeError("Could not build context for event.")


def generate_commentary_from_event(
    *,
    checkpoint: str | Path,
    event: dict[str, Any],
    event_type: str | None = None,
    context: dict[str, Any] | None = None,
    max_new_tokens: int = 48,
    num_beams: int = 4,
    temperature: float = 1.0,
) -> str:
    """
    Generate one commentary line from an event row and a fine-tuned model.
    """
    AutoModelForSeq2SeqLM, AutoTokenizer = _require_transformers()

    checkpoint = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if event_type is None:
        event_type = classify_event(event)

    prompt = prompt_from_match_event(event, event_type, context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.15,
    }

    if temperature and temperature != 1.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _event_from_manual_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "innings": args.innings,
        "over": args.over,
        "ball_in_over": args.ball_in_over,
        "batter": args.batter,
        "bowler": args.bowler,
        "runs_off_bat": args.runs,
        "extras": args.extras,
        "wicket_type": "" if args.wicket == "none" else args.wicket,
        "player_dismissed": args.player_dismissed or args.batter,
        "wides": 0,
        "noballs": 0,
        "byes": 0,
        "legbyes": 0,
    }


def _load_event_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    if args.event_json:
        event = json.loads(args.event_json)
        event_type = args.event_type or classify_event(event)
        return event, event_type, None

    if args.match:
        events = load_match_events(args.match)
        event = events[args.event_index]
        event_type = args.event_type or classify_event(event)
        context = build_context_for_event(events, args.event_index)
        return event, event_type, context

    event = _event_from_manual_args(args)
    event_type = args.event_type or classify_event(event)
    context = {
        "innings_score": args.score_runs,
        "innings_wickets": args.score_wickets,
    } if args.score_runs is not None and args.score_wickets is not None else None
    return event, event_type, context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cricket commentary with a fine-tuned T5 checkpoint."
    )
    parser.add_argument("--checkpoint", default="models/t5-cricket-commentary")
    parser.add_argument("--match", default="", help="Optional Cricsheet match JSON path.")
    parser.add_argument("--event-index", type=int, default=0, help="0-based event index when --match is used.")
    parser.add_argument("--event-json", default="", help="Optional event row as a JSON string.")
    parser.add_argument("--event-type", default="", help="Override event type. Otherwise classify_event is used.")
    parser.add_argument("--batter", default="Mohammad Rizwan")
    parser.add_argument("--bowler", default="Mohammad Ali")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--extras", type=int, default=0)
    parser.add_argument("--wicket", default="none")
    parser.add_argument("--player-dismissed", default="")
    parser.add_argument("--innings", type=int, default=1)
    parser.add_argument("--over", type=int, default=0)
    parser.add_argument("--ball-in-over", type=int, default=3)
    parser.add_argument("--score-runs", type=int, default=None)
    parser.add_argument("--score-wickets", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    event, event_type, context = _load_event_from_args(args)

    prompt = prompt_from_match_event(event, event_type, context)
    if args.show_prompt:
        print("Prompt:\n")
        print(prompt)
        print("\nGenerated commentary:\n")

    commentary = generate_commentary_from_event(
        checkpoint=args.checkpoint,
        event=event,
        event_type=event_type,
        context=context,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )
    print(commentary)


if __name__ == "__main__":
    main()
