from load_events import load_match_events
from classify_event import classify_event
from load_commentary import build_commentary_bank
from retriever import get_commentary_examples
from generator import generate_commentary


def run_match_demo(match_path, commentary_csv_path, max_events=None):
    """
    Run a sequential commentary demo for one match file.
    """
    events = load_match_events(match_path)
    commentary_bank = build_commentary_bank(commentary_csv_path)

    if max_events is not None:
        events = events[:max_events]

    for event in events:
        event_type = classify_event(event)
        examples = get_commentary_examples(commentary_bank, event_type, k=3)
        commentary = generate_commentary(event, event_type, examples)

        print(
            f"Innings {event['innings']} | "
            f"{event['over']}.{event['ball_in_over']} | "
            f"{event['batter']} vs {event['bowler']} -> {event_type}"
        )
        print(f"Commentary: {commentary}")
        print("-" * 70)


if __name__ == "__main__":
    run_match_demo("raw/1527574.json", "raw/train.csv", max_events=24)