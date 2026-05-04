import argparse
import time
from pathlib import Path

from load_soccer_events import load_soccer_match_events
from classify_soccer_event import classify_soccer_event
from load_soccer_commentary import build_soccer_commentary_bank
from retriever import get_commentary_examples
from soccer_generator import generate_soccer_commentary


DEFAULT_COMMENTARY_FILE = "commentary_train_augmented.csv"


INTERESTING_EVENT_TYPES = {
    "goal",
    "shot",
    "save",
    "corner",
    "free_kick",
    "foul",
    "yellow_card",
    "red_card",
    "offside",
    "substitution",
}


def _default_soccer_commentary_path(filename):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates = [
        project_root / "raw_soccer" / "yallashoot" / filename,
        Path.cwd() / "raw_soccer" / "yallashoot" / filename,
        script_dir / filename,
        Path.cwd() / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return project_root / "raw_soccer" / "yallashoot" / filename


def _default_soccer_match_path():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates = [
        project_root / "raw_soccer" / "statsbomb" / "events",
        Path.cwd() / "raw_soccer" / "statsbomb" / "events",
    ]

    for folder in candidates:
        if folder.exists():
            json_files = sorted(folder.glob("*.json"))
            if json_files:
                return json_files[0]

    raise FileNotFoundError("No StatsBomb event JSON files found in raw_soccer/statsbomb/events")


def _clock_label(event):
    minute = int(event.get("minute", 0) or 0)
    second = int(event.get("second", 0) or 0)
    return f"{minute}:{second:02d}"


def _print_demo_header(match_path, commentary_csv_path, max_events, debug, include_passes):
    print("\nReal-Time Soccer Commentary Demo")
    print(f"Match data: {Path(match_path).name}")
    print(f"Commentary corpus: {Path(commentary_csv_path).name}")

    if max_events is not None:
        print(f"Showing first {max_events} relevant events")

    print("Mode: debug" if debug else "Mode: final demo")
    print("Including normal passes." if include_passes else "Skipping normal pass events.")
    print("=" * 72)


def _print_clean_event(event, commentary):
    clock = _clock_label(event)
    team = event.get("team", "")
    player = event.get("player", "")

    print(f"{clock:<8} {team} | {player}")
    print(f"         {commentary}")


def _print_debug_event(event, event_type, commentary, examples, scoreline):
    print(
        f"{_clock_label(event)} | "
        f"{event.get('team', '')} | "
        f"{event.get('player', '')} -> {event_type}"
    )
    if scoreline:
        print(f"Scoreline: {scoreline}")
    print(f"Commentary: {commentary}")

    if examples:
        print("Top retrieved examples:")
        for idx, example in enumerate(examples[:3], start=1):
            print(f"  {idx}. {example}")

    print("-" * 72)


def _scoreline_text(team_order, team_scores):
    if len(team_order) < 2:
        return ""

    left = team_order[0]
    right = team_order[1]
    return f"{left} {team_scores.get(left, 0)} - {team_scores.get(right, 0)} {right}"


def run_soccer_demo(
    match_path,
    commentary_csv_path,
    max_events=20,
    debug=False,
    delay_seconds=0.0,
    include_passes=False,
):
    events = load_soccer_match_events(match_path)
    commentary_bank = build_soccer_commentary_bank(commentary_csv_path)

    _print_demo_header(match_path, commentary_csv_path, max_events, debug, include_passes)

    shown = 0
    team_order = []
    team_scores = {}

    for event in events:
        team = event.get("team", "")
        if team and team not in team_scores:
            team_scores[team] = 0
            team_order.append(team)

        event_type = classify_soccer_event(event)

        if event_type == "other":
            continue

        if event_type == "pass" and not include_passes:
            continue

        if max_events is not None and shown >= max_events:
            break

        if event_type == "goal" and team:
            team_scores[team] += 1

        scoreline = _scoreline_text(team_order, team_scores)
        context = {
            "goal_scored": event_type == "goal",
            "scoreline": scoreline,
        }

        examples = get_commentary_examples(commentary_bank, event_type, event=event, k=3)
        commentary = generate_soccer_commentary(event, event_type, examples, context)

        if debug:
            _print_debug_event(event, event_type, commentary, examples, scoreline)
        else:
            _print_clean_event(event, commentary)

        shown += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("\nEnd of soccer demo.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stream a clean soccer commentary demo from StatsBomb event data."
    )
    parser.add_argument(
        "--match",
        default=str(_default_soccer_match_path()),
        help="Path to a StatsBomb event JSON file.",
    )
    parser.add_argument(
        "--commentary-csv",
        default=str(_default_soccer_commentary_path(DEFAULT_COMMENTARY_FILE)),
        help="Path to the soccer commentary CSV.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=20,
        help="Maximum number of relevant events to show. Use 0 to show all relevant events.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show labels, retrieved examples, and scoreline trace output.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to pause between events for a live-demo effect.",
    )
    parser.add_argument(
        "--include-passes",
        action="store_true",
        help="Include regular pass events in final demo mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    max_events = None if args.max_events == 0 else args.max_events

    run_soccer_demo(
        args.match,
        args.commentary_csv,
        max_events=max_events,
        debug=args.debug,
        delay_seconds=args.delay,
        include_passes=args.include_passes,
    )