SOCCER_EVENT_TYPES = [
    "pass",
    "shot",
    "goal",
    "save",
    "corner",
    "free_kick",
    "foul",
    "yellow_card",
    "red_card",
    "offside",
    "substitution",
    "other",
]


def classify_soccer_event(event):
    """
    Map one flattened soccer event into a small fixed soccer label set.
    """
    raw = str(event.get("event_type_raw", "")).strip().lower()
    pass_type = str(event.get("pass_type", "")).strip().lower()
    shot_outcome = str(event.get("shot_outcome", "")).strip().lower()
    foul_card = str(event.get("foul_card", "")).strip().lower()
    goalkeeper_type = str(event.get("goalkeeper_type", "")).strip().lower()

    # Goals and shots
    if raw == "shot" and shot_outcome == "goal":
        return "goal"
    if raw == "shot":
        return "shot"

    # Goalkeeper events
    if raw in {"goal keeper", "goalkeeper"}:
        if goalkeeper_type == "shot faced":
            return "save"
        return "save"

    # Passing / set pieces
    if raw == "pass" and pass_type == "corner":
        return "corner"
    if raw == "pass" and pass_type == "free kick":
        return "free_kick"
    if raw == "pass":
        return "pass"

    # Fouls / cards
    if raw == "foul committed":
        if foul_card == "yellow card":
            return "yellow_card"
        if foul_card in {"red card", "second yellow"}:
            return "red_card"
        return "foul"

    if raw == "offside":
        return "offside"

    if raw == "substitution":
        return "substitution"

    return "other"