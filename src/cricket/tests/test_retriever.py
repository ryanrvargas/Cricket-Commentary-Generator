from cricket.load_commentary import build_commentary_bank
from cricket.load_events import load_match_events
from cricket.classify_event import classify_event
from common.retriever import get_commentary_examples

bank = build_commentary_bank("raw/train.csv")

# Use a real match file that has a good variety of events
events = load_match_events("raw/1527575.json")

test_labels = [
    "boundary_four",
    "wicket_caught",
    "single",
    "wide",
    "bye_or_legbye"
]

for label in test_labels:
    print(f"\n=== {label} ===")

    # Find the first real event in the match that matches this label
    matching_event = None
    for event in events:
        if classify_event(event) == label:
            matching_event = event
            break

    if matching_event is None:
        print("No matching event found in this match.")
        continue

    print(
        f"Using event: innings={matching_event['innings']}, "
        f"over={matching_event['over']}.{matching_event['ball_in_over']}, "
        f"batter={matching_event['batter']}, "
        f"bowler={matching_event['bowler']}"
    )

    examples = get_commentary_examples(bank, label, event=matching_event, k=3)

    for i, example in enumerate(examples, start=1):
        print(f"{i}. {example}")