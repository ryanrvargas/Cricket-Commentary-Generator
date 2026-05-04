
"""
classify_event.py
------------------
This module provides functionality to classify a single ball-by-ball delivery row from Cricsheet (or similar cricket data sources)
into a fixed set of cricket event labels. The classification is based on the values of wicket type, runs, and extras in the row.

Functions:
    - classify_event(row): Classifies a delivery row into a specific cricket event label.
    - _to_int(value): Safely converts various CSV cell values to an integer, treating blanks as zero.

Event labels returned include:
    - wicket_bowled, wicket_caught, wicket_lbw, run_out
    - wide, no_ball, bye_or_legbye
    - boundary_six, boundary_four, triple, double, single, dot_ball
    - other (for any unclassified event)

Typical usage:
    event = classify_event(row)
"""

def _to_int(value):
    """
    Safely convert a value from a CSV row to an integer.

    Handles cases where the value may be None, an empty string, or already an integer.
    Blank or None values are treated as 0.

    Args:
        value: The value to convert (can be str, int, None, etc.).

    Returns:
        int: The integer representation of the value, or 0 if blank/None.
    """
    if value is None:
        return 0
    text = str(value).strip()
    if text == "":
        return 0
    return int(text)


def classify_event(row):
    """
    Classify a single Cricsheet delivery row into a fixed cricket event label.

    The function inspects the wicket type, runs, and extras in the row to determine
    the most appropriate event label. The classification order is:
        1. Wicket events (bowled, caught, lbw, run out)
        2. Extras (wide, no ball, byes/legbyes)
        3. Scoring shots (6, 4, 3, 2, 1, dot ball)
        4. Other (if none of the above apply)

    Args:
        row (dict): A dictionary representing a single ball delivery, with keys such as
            'wicket_type', 'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes'.

    Returns:
        str: The classified event label (e.g., 'wicket_bowled', 'wide', 'boundary_six', etc.).
    """
    wicket_type = str(row.get("wicket_type", "")).strip().lower()

    runs_off_bat = _to_int(row.get("runs_off_bat"))
    extras = _to_int(row.get("extras"))
    wides = _to_int(row.get("wides"))
    noballs = _to_int(row.get("noballs"))
    byes = _to_int(row.get("byes"))
    legbyes = _to_int(row.get("legbyes"))


    # Wicket events
    if wicket_type == "bowled":
        return "wicket_bowled"
    if wicket_type in {"caught", "caught and bowled"}:
        return "wicket_caught"
    if wicket_type == "lbw":
        return "wicket_lbw"
    if wicket_type == "run out":
        return "run_out"


    # Extras
    if wides > 0:
        return "wide"
    if noballs > 0:
        return "no_ball"
    if byes > 0 or legbyes > 0:
        return "bye_or_legbye"


    # Normal scoring shots
    if runs_off_bat == 6:
        return "boundary_six"
    if runs_off_bat == 4:
        return "boundary_four"
    if runs_off_bat == 3:
        return "triple"
    if runs_off_bat == 2:
        return "double"
    if runs_off_bat == 1:
        return "single"
    if runs_off_bat == 0 and extras == 0:
        return "dot_ball"

    # Fallback for any unclassified event
    return "other"