"""
Map one Cricsheet ball-by-ball row to one fixed event label.
"""

def _to_int(value):
    """
    Convert CSV values like '', None, '0', or 0 into an int safely.
    Blank values become 0.
    """
    if value is None:
        return 0
    text = str(value).strip()
    if text == "":
        return 0
    return int(text)


def classify_event(row):
    """
    Classify a single Cricsheet delivery row into one of the project's
    fixed cricket event labels.
    """
    wicket_type = str(row.get("wicket_type", "")).strip().lower()

    runs_off_bat = _to_int(row.get("runs_off_bat"))
    extras = _to_int(row.get("extras"))
    wides = _to_int(row.get("wides"))
    noballs = _to_int(row.get("noballs"))
    byes = _to_int(row.get("byes"))
    legbyes = _to_int(row.get("legbyes"))

    # Wickets first
    if wicket_type == "bowled":
        return "wicket_bowled"
    if wicket_type in {"caught", "caught and bowled"}:
        return "wicket_caught"
    if wicket_type == "lbw":
        return "wicket_lbw"
    if wicket_type == "run out":
        return "run_out"

    # Extras next
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

    return "other"