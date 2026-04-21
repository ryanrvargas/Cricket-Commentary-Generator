from load_events import load_match_events
from classify_event import classify_event

match_path = "raw/1527574.json"   # change if needed

events = load_match_events(match_path)

for event in events[:12]:
    label = classify_event(event)
    print(
        f"Innings {event['innings']} | "
        f"{event['over']}.{event['ball_in_over']} "
        f"(delivery #{event['delivery_index']}) | "
        f"{event['batter']} vs {event['bowler']} -> {label}"
    )

print("\n... (Testing edge cases) ...\n")
for event in events:
    if event["innings"] == 1 and event["over"] == 4:
        label = classify_event(event)
        print(
            f"Innings {event['innings']} | "
            f"{event['over']}.{event['ball_in_over']} "
            f"(delivery #{event['delivery_index']}) | "
            f"{event['batter']} vs {event['bowler']} -> {label}"
        )