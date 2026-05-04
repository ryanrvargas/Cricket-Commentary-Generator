"""
t5_stream_demo.py
-----------------
Run a sequential cricket commentary demo using a fine-tuned T5/FLAN-T5 checkpoint.

This is separate from stream_demo.py on purpose:
- stream_demo.py is the polished TF-IDF + template demo.
- t5_stream_demo.py is the experimental neural model demo.

Example:
    python src/t5_stream_demo.py --checkpoint models/t5-cricket-commentary-balanced-fast --max-events 24
"""

from __future__ import annotations

import argparse
import sys
import time
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
from model_infer import enforce_event_fact  # noqa: E402


DEFAULT_MATCH_FILE = "1527575.json"
DEFAULT_CHECKPOINT = "models/t5-cricket-commentary-balanced-fast"


def _require_transformers():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing T5 demo dependencies. Install them with:\n"
            "pip install torch transformers sentencepiece"
        ) from exc

    return AutoModelForSeq2SeqLM, AutoTokenizer


def _default_match_path(filename: str) -> Path:
    """
    Find the default match file whether the script is run from the project root
    or from inside src/.
    """
    candidates = [
        PROJECT_ROOT / "raw" / filename,
        Path.cwd() / "raw" / filename,
        CURRENT_DIR / filename,
        Path.cwd() / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return PROJECT_ROOT / "raw" / filename


def _delivery_label(event: dict[str, Any]) -> str:
    """
    Format the ball number for display.
    """
    over = event.get("over", 0)
    ball = event.get("ball_in_over", 0)

    if ball == 0:
        return f"{over}.extra"

    return f"{over}.{ball}"


def _build_event_context(
    event: dict[str, Any],
    runs_on_ball: int,
    wickets_lost_on_ball: int,
    innings_score: int,
    innings_wickets: int,
) -> dict[str, Any]:
    """
    Keep a small score context for compatibility with prompt builders.
    The current simplified T5 prompt does not use score context, but this keeps
    the demo ready for future prompt experiments.
    """
    return {
        "innings": event.get("innings"),
        "over": event.get("over"),
        "ball_in_over": event.get("ball_in_over"),
        "runs_on_ball": runs_on_ball,
        "wickets_lost_on_ball": wickets_lost_on_ball,
        "innings_score": innings_score,
        "innings_wickets": innings_wickets,
        "is_end_of_over": event.get("ball_in_over") == 6,
    }


def _load_t5(checkpoint: str | Path, device_override: str = ""):
    """
    Load the tokenizer and model once for the whole stream.
    This is much faster than calling model_infer.py separately for every ball.
    """
    AutoModelForSeq2SeqLM, AutoTokenizer = _require_transformers()

    checkpoint = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    if device_override:
        device = device_override
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


def _generate_with_loaded_model(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
) -> str:
    """
    Generate one commentary line using an already-loaded seq2seq model.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
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


def _print_header(match_path: str | Path, checkpoint: str | Path, max_events: int | None, fact_guard: bool) -> None:
    print("\nT5 Cricket Commentary Demo")
    print(f"Match data: {Path(match_path).name}")
    print(f"Checkpoint: {checkpoint}")

    if max_events is None:
        print("Showing full match")
    else:
        print(f"Showing first {max_events} deliveries")

    print(f"Fact guard: {'on' if fact_guard else 'off'}")
    print("=" * 72)


def _print_innings_header(innings_number: int, batting_team: str) -> None:
    team_text = f" - {batting_team}" if batting_team else ""
    print(f"\nInnings {innings_number}{team_text}")
    print("-" * 72)


def _print_delivery(event: dict[str, Any], event_type: str, commentary: str, show_event_type: bool) -> None:
    ball = _delivery_label(event)
    batter = event.get("batter", "")
    bowler = event.get("bowler", "")

    if show_event_type:
        print(f"{ball:<8} {batter} vs {bowler} -> {event_type}")
    else:
        print(f"{ball:<8} {batter} vs {bowler}")

    print(f"         {commentary}")


def run_t5_demo(
    *,
    checkpoint: str | Path,
    match_path: str | Path,
    max_events: int | None = 24,
    delay_seconds: float = 0.0,
    fact_guard: bool = True,
    show_prompts: bool = False,
    show_event_type: bool = False,
    max_new_tokens: int = 48,
    num_beams: int = 4,
    temperature: float = 1.0,
    device_override: str = "",
) -> None:
    """
    Run a sequential T5 commentary demo over a saved Cricsheet match file.
    """
    events = load_match_events(match_path)
    if max_events is not None:
        events = events[:max_events]

    print("Loading T5 model...")
    tokenizer, model, device = _load_t5(checkpoint, device_override=device_override)
    print(f"Model loaded on {device}.")

    _print_header(match_path, checkpoint, max_events, fact_guard)

    current_innings = None
    innings_score = 0
    innings_wickets = 0

    for event in events:
        if current_innings != event.get("innings"):
            current_innings = event.get("innings")
            innings_score = 0
            innings_wickets = 0
            _print_innings_header(current_innings, event.get("batting_team", ""))

        event_type = classify_event(event)

        runs_on_ball = int(event.get("runs_off_bat", 0) or 0) + int(event.get("extras", 0) or 0)
        wickets_lost_on_ball = 1 if event.get("wicket_type") else 0

        innings_score += runs_on_ball
        innings_wickets += wickets_lost_on_ball

        context = _build_event_context(
            event,
            runs_on_ball,
            wickets_lost_on_ball,
            innings_score,
            innings_wickets,
        )

        prompt = prompt_from_match_event(event, event_type, context)

        if show_prompts:
            print("\nPrompt:")
            print(prompt)
            print()

        commentary = _generate_with_loaded_model(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
        )

        if fact_guard:
            commentary = enforce_event_fact(commentary, event, event_type)

        _print_delivery(event, event_type, commentary, show_event_type)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("\nEnd of T5 demo.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a cricket commentary demo using a fine-tuned T5 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the fine-tuned T5/FLAN-T5 checkpoint directory.",
    )
    parser.add_argument(
        "--match",
        default=str(_default_match_path(DEFAULT_MATCH_FILE)),
        help="Path to a Cricsheet match JSON file.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=24,
        help="Maximum number of deliveries to show. Use 0 to show the full match.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to pause between deliveries for a live-demo effect.",
    )
    parser.add_argument(
        "--no-fact-guard",
        action="store_true",
        help="Show raw T5 output without event fact guardrails.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print the model prompt before each generated line.",
    )
    parser.add_argument(
        "--show-event-type",
        action="store_true",
        help="Print the classified event type beside each delivery.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum number of new tokens to generate per delivery.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam count for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Values other than 1.0 enable sampling.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional device override, such as cpu or cuda.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    max_events = None if args.max_events == 0 else args.max_events

    run_t5_demo(
        checkpoint=args.checkpoint,
        match_path=args.match,
        max_events=max_events,
        delay_seconds=args.delay,
        fact_guard=not args.no_fact_guard,
        show_prompts=args.show_prompts,
        show_event_type=args.show_event_type,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
