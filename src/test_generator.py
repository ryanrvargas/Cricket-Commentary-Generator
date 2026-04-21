from load_events import load_match_events
from classify_event import classify_event
from load_commentary import build_commentary_bank
from retriever import get_commentary_examples
from generator import generate_commentary

match_path = "raw/1527574.json"

events = load_match_events(match_path)
bank = build_commentary_bank("raw/train.csv")

for event in events[:12]:
    label = classify_event(event)
    examples = get_commentary_examples(bank, label, k=3)
    commentary = generate_commentary(event, label, examples)

    print(
        f"Innings {event['innings']} | "
        f"{event['over']}.{event['ball_in_over']} | "
        f"{event['batter']} vs {event['bowler']} -> {label}"
    )
    print(f"Generated: {commentary}")
    print()