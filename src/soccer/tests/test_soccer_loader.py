from pathlib import Path
from soccer.load_soccer_events import load_soccer_match_events

events_dir = Path("raw_soccer/statsbomb/events")

# grab the first json file you have
json_files = sorted(events_dir.glob("*.json"))
if not json_files:
    raise FileNotFoundError("No StatsBomb event JSON files found in raw_soccer/statsbomb/events")

match_path = json_files[0]
events = load_soccer_match_events(match_path)

print(f"Loaded file: {match_path.name}")
print(f"Total events: {len(events)}")
print()

for event in events[:10]:
    print(event)