import json
from pathlib import Path


def _safe_name(obj, *keys):
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, {})
    if isinstance(current, dict):
        return current.get("name", "")
    return current or ""


def load_soccer_match_events(json_path):
    """
    Load a StatsBomb events JSON file and flatten each event into a simple dict.
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    match_id = json_path.stem
    flattened = []

    for idx, event in enumerate(events):
        event_type_raw = _safe_name(event, "type")
        team = _safe_name(event, "team")
        player = _safe_name(event, "player")
        period = event.get("period", 0)
        minute = event.get("minute", 0)
        second = event.get("second", 0)

        pass_type = _safe_name(event, "pass", "type")
        shot_outcome = _safe_name(event, "shot", "outcome")
        shot_type = _safe_name(event, "shot", "type")
        foul_card = _safe_name(event, "foul_committed", "card")
        substitution_replacement = _safe_name(event, "substitution", "replacement")
        goalkeeper_outcome = _safe_name(event, "goalkeeper", "outcome")

        flattened.append(
            {
                "sport": "soccer",
                "match_id": match_id,
                "event_index": idx,
                "period": period,
                "minute": minute,
                "second": second,
                "team": team,
                "player": player,
                "event_type_raw": event_type_raw,
                "pass_type": pass_type,
                "shot_outcome": shot_outcome,
                "shot_type": shot_type,
                "foul_card": foul_card,
                "substitution_replacement": substitution_replacement,
                "goalkeeper_outcome": goalkeeper_outcome,
            }
        )

    return flattened