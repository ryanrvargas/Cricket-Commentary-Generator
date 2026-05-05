import argparse
from pathlib import Path

from src.cricket.load_commentary import build_commentary_bank
from src.cricket.classify_event import classify_event
from src.cricket.generator import generate_commentary
from src.common.retriever import get_commentary_examples
from src.common.providers import (
    ProviderNotConfiguredError,
    build_provider,
    iter_limited_events,
)

DEFAULT_MATCH_FILE = "1527575.json"
DEFAULT_COMMENTARY_FILE = "train.csv"


def _default_data_path(filename):
    """
    Find a default data file whether the script is run from the project root,
    from src/, or from a package folder like src/cricket/.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    candidates = [
        project_root / "raw" / filename,
        Path.cwd() / "raw" / filename,
        script_dir / filename,
        Path.cwd() / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return project_root / "raw" / filename


def build_event_context(event, runs_on_ball, wickets_lost_on_ball, innings_score, innings_wickets):
    """
    Build a small context dictionary for one delivery.
    """
    return {
        "innings": event["innings"],
        "over": event["over"],
        "ball_in_over": event["ball_in_over"],
        "runs_on_ball": runs_on_ball,
        "wickets_lost_on_ball": wickets_lost_on_ball,
        "innings_score": innings_score,
        "innings_wickets": innings_wickets,
        "is_end_of_over": event["ball_in_over"] == 6,
    }


def _delivery_label(event):
    """
    Format the ball number for display.

    Legal balls show as over.ball, such as 4.6.
    Illegal deliveries before a legal ball show as over.extra.
    """
    over = event.get("over", 0)
    ball = event.get("ball_in_over", 0)

    if ball == 0:
        return f"{over}.extra"

    return f"{over}.{ball}"


def _print_demo_header(
    match_path,
    commentary_csv_path,
    max_events,
    debug,
    provider_name,
    delay_seconds,
):
    """
    Print a clean title block for the demo.
    """
    print("\nReal-Time Cricket Commentary Demo")
    print(f"Match data: {Path(match_path).name}")
    print(f"Commentary corpus: {Path(commentary_csv_path).name}")
    print(f"Provider: {provider_name}")

    if provider_name == "timed" or delay_seconds > 0:
        print(f"Pseudo-live delay: {delay_seconds:.2f}s")

    if max_events is not None:
        print(f"Showing first {max_events} deliveries")

    print("Mode: debug" if debug else "Mode: final demo")
    print("=" * 72)


def _print_innings_header(innings_number, batting_team):
    """
    Print an innings divider that looks good in a final demo.
    """
    team_text = f" - {batting_team}" if batting_team else ""
    print(f"\nInnings {innings_number}{team_text}")
    print("-" * 72)


def _print_clean_delivery(event, commentary):
    """
    Print one presentation-ready commentary line.
    """
    ball = _delivery_label(event)
    batter = event.get("batter", "")
    bowler = event.get("bowler", "")

    print(f"{ball:<8} {batter} vs {bowler}")
    print(f"         {commentary}")


def _print_debug_delivery(event, event_type, context, commentary):
    """
    Print trace-style details when debugging.
    """
    print(
        f"Innings {event['innings']} | "
        f"{_delivery_label(event)} | "
        f"{event['batter']} vs {event['bowler']} -> {event_type}"
    )
    print(f"Provider: {event.get('_provider', 'unknown')}")
    print(f"Source: {event.get('_provider_source', 'unknown')}")
    print(f"Sequence: {event.get('_provider_sequence', 'unknown')}")
    print(f"Score after ball: {context['innings_score']}/{context['innings_wickets']}")
    print(f"Runs on ball: {context['runs_on_ball']}")
    print(f"Commentary: {commentary}")
    print("-" * 72)


def run_match_demo(
    match_path,
    commentary_csv_path,
    max_events=None,
    debug=False,
    delay_seconds=0.0,
    provider_name="historical",
):
    """
    Run a sequential cricket commentary demo and track live score.

    Provider modes:
        historical - read saved Cricsheet events and emit immediately
        timed      - read saved Cricsheet events with delay between events
        live       - future live API slot, currently not configured
    """
    commentary_bank = build_commentary_bank(commentary_csv_path)

    effective_provider = provider_name
    if provider_name == "historical" and delay_seconds > 0:
        effective_provider = "timed"

    provider = build_provider(
        sport="cricket",
        provider_name=effective_provider,
        source_path=match_path,
        delay_seconds=delay_seconds,
    )

    _print_demo_header(
        match_path,
        commentary_csv_path,
        max_events,
        debug,
        effective_provider,
        delay_seconds,
    )

    current_innings = None
    innings_score = 0
    innings_wickets = 0

    try:
        for event in iter_limited_events(provider, max_events=max_events):
            if current_innings != event["innings"]:
                current_innings = event["innings"]
                innings_score = 0
                innings_wickets = 0
                _print_innings_header(current_innings, event.get("batting_team", ""))

            event_type = classify_event(event)
            examples = get_commentary_examples(commentary_bank, event_type, event=event, k=3)

            runs_on_ball = event["runs_off_bat"] + event["extras"]
            wickets_lost_on_ball = 1 if event["wicket_type"] else 0

            innings_score += runs_on_ball
            innings_wickets += wickets_lost_on_ball

            context = build_event_context(
                event,
                runs_on_ball,
                wickets_lost_on_ball,
                innings_score,
                innings_wickets,
            )

            commentary = generate_commentary(event, event_type, examples, context)

            if debug:
                _print_debug_delivery(event, event_type, context, commentary)
            else:
                _print_clean_delivery(event, commentary)

    except ProviderNotConfiguredError as exc:
        print(str(exc))

    print("\nEnd of demo.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stream a clean cricket commentary demo from an event provider."
    )
    parser.add_argument(
        "--match",
        default=str(_default_data_path(DEFAULT_MATCH_FILE)),
        help="Path to a Cricsheet match JSON file.",
    )
    parser.add_argument(
        "--commentary-csv",
        default=str(_default_data_path(DEFAULT_COMMENTARY_FILE)),
        help="Path to the commentary CSV corpus.",
    )
    parser.add_argument(
        "--provider",
        choices=["historical", "timed", "live"],
        default="historical",
        help="Event provider mode.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=24,
        help="Maximum number of deliveries to show. Use 0 to show the full match.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show event labels, provider info, score details, and runs-on-ball trace output.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to pause between events for timed pseudo-live replay.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    max_events = None if args.max_events == 0 else args.max_events

    run_match_demo(
        args.match,
        args.commentary_csv,
        max_events=max_events,
        debug=args.debug,
        delay_seconds=args.delay,
        provider_name=args.provider,
    )