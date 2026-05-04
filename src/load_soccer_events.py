import json
from pathlib import Path


def _safe_name(obj, *keys):
    """
    Walk nested dictionaries and return a nested 'name' value when present.
    Example:
        _safe_name(event, "type") -> "Pass"
        _safe_name(event, "pass", "type") -> "Corner"
    """
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, {})
    if isinstance(current, dict):
        return current.get("name", "")
    return current or ""


def _safe_value(obj, *keys, default=""):
    """
    Walk nested dictionaries and return the raw final value.
    """
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def load_soccer_match_events(json_path):
    """
    Load a StatsBomb events JSON file and flatten each event into a simpler dict.

    This file is the soccer equivalent of load_events.py in the cricket pipeline.
    It keeps only the fields most useful for classification, retrieval, and demo output.
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    match_id = json_path.stem
    flattened = []

    for fallback_index, event in enumerate(events, start=1):
        event_type_raw = _safe_name(event, "type")
        team = _safe_name(event, "team")
        player = _safe_name(event, "player")
        position = _safe_name(event, "position")

        minute = event.get("minute", 0)
        second = event.get("second", 0)
        period = event.get("period", 0)
        timestamp = event.get("timestamp", "")
        event_index = event.get("index", fallback_index)

        pass_type = _safe_name(event, "pass", "type")
        pass_outcome = _safe_name(event, "pass", "outcome")
        pass_height = _safe_name(event, "pass", "height")
        pass_recipient = _safe_name(event, "pass", "recipient")

        shot_outcome = _safe_name(event, "shot", "outcome")
        shot_type = _safe_name(event, "shot", "type")
        shot_xg = _safe_value(event, "shot", "statsbomb_xg", default=0.0)

        goalkeeper_type = _safe_name(event, "goalkeeper", "type")
        goalkeeper_outcome = _safe_name(event, "goalkeeper", "outcome")

        foul_card = _safe_name(event, "foul_committed", "card")
        substitution_replacement = _safe_name(event, "substitution", "replacement")

        flattened.append(
            {
                "sport": "soccer",
                "match_id": match_id,
                "event_index": event_index,
                "period": period,
                "timestamp": timestamp,
                "minute": minute,
                "second": second,
                "team": team,
                "player": player,
                "position": position,
                "event_type_raw": event_type_raw,
                "play_pattern": _safe_name(event, "play_pattern"),
                "possession_team": _safe_name(event, "possession_team"),
                "pass_type": pass_type,
                "pass_outcome": pass_outcome,
                "pass_height": pass_height,
                "pass_recipient": pass_recipient,
                "shot_outcome": shot_outcome,
                "shot_type": shot_type,
                "shot_xg": shot_xg,
                "goalkeeper_type": goalkeeper_type,
                "goalkeeper_outcome": goalkeeper_outcome,
                "foul_card": foul_card,
                "substitution_replacement": substitution_replacement,
                "under_pressure": bool(event.get("under_pressure", False)),
                # Compatibility aliases so the current cricket-shaped retriever can still rank.
                "batter": player,
                "bowler": team,
                "runs_off_bat": 1 if str(shot_outcome).strip().lower() == "goal" else 0,
                "wicket_type": "",
                "over": minute,
            }
        )

    return flattened


if __name__ == "__main__":
    sample_path = Path("raw_soccer/statsbomb/events")
    json_files = sorted(sample_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("No StatsBomb event files found in raw_soccer/statsbomb/events")

    events = load_soccer_match_events(json_files[0])
    print(f"Loaded file: {json_files[0].name}")
    print(f"Total events: {len(events)}")
    print()

    for event in events[:10]:
        print(event)