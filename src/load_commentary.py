# loads the Kaggle train.cvs
# Parses the single rows column into usable fields
# Maps each commentary line into your fixed event labels

import re
import pandas as pd

from event_types import EVENT_TYPES


def _clean_text(text):
    """
    Normalize whitespace and return a stripped string.
    """
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract(pattern, text, default=""):
    """
    Extract the first regex group from text. If no match is found,
    return the default value.
    """
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _clean_text(match.group(1))
    return default


def _to_int(value, default=0):
    """
    Safely convert a parsed string value to int.
    """
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def parse_commentary_row(raw_row):
    """
    Parse one Kaggle commentary row from the single 'rows' column into
    a structured dictionary.

    The dataset stores metadata and commentary in one long text string:
    <start_of_table> ... <end_of_table> commentary ...
    """
    raw_row = str(raw_row)

    play_type = _extract(r"play type description is (.*?) batting team is ", raw_row)
    batting_team = _extract(r"batting team is (.*?) bowling team is ", raw_row)
    bowling_team = _extract(r"bowling team is (.*?) total runs on delivery is ", raw_row)
    total_runs = _to_int(_extract(r"total runs on delivery is (.*?) bowler name is ", raw_row, "0"))
    bowler_name = _extract(r"bowler name is (.*?) batsman name is ", raw_row)
    batsman_name = _extract(r"batsman name is (.*?) over runs is ", raw_row)
    over_runs = _to_int(_extract(r"over runs is (.*?) dismissal is ", raw_row, "0"))
    dismissal_flag = _extract(r"dismissal is (.*?) dismissal type is ", raw_row).lower() == "true"
    dismissal_type = _extract(r"dismissal type is (.*?) dismissal text is ", raw_row)
    innings_wickets = _to_int(_extract(r"innings wickets is (.*?) <end_of_table>", raw_row, "0"))

    # Everything after <end_of_table> is the actual commentary sentence.
    commentary = raw_row.split("<end_of_table>", 1)[-1]
    commentary = re.sub(r"^\s*commentary\s*", "", commentary, flags=re.IGNORECASE)
    commentary = _clean_text(commentary)

    return {
        "raw_text": raw_row,
        "play_type": play_type.lower(),
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "total_runs": total_runs,
        "bowler_name": bowler_name,
        "batsman_name": batsman_name,
        "over_runs": over_runs,
        "dismissal_flag": dismissal_flag,
        "dismissal_type": dismissal_type.lower(),
        "innings_wickets": innings_wickets,
        "commentary": commentary,
    }


def map_commentary_to_event_type(parsed_row):
    """
    Map a parsed commentary row into one of the project's fixed event labels.

    This does NOT define match truth. It only groups commentary lines into
    useful buckets for later retrieval.
    """
    play_type = parsed_row["play_type"]
    total_runs = parsed_row["total_runs"]
    dismissal_type = parsed_row["dismissal_type"]

    # Dismissals first
    if dismissal_type == "bowled":
        return "wicket_bowled"
    if dismissal_type == "caught":
        return "wicket_caught"
    if dismissal_type == "leg before wicket":
        return "wicket_lbw"
    if dismissal_type == "run out":
        return "run_out"

    # Extras
    if play_type == "wide":
        return "wide"
    if play_type == "no ball":
        return "no_ball"
    if play_type in {"bye", "leg bye"}:
        return "bye_or_legbye"

    # Boundaries
    if play_type == "four":
        return "boundary_four"
    if play_type == "six":
        return "boundary_six"

    # Standard scoring plays
    if play_type == "run":
        if total_runs == 1:
            return "single"
        if total_runs == 2:
            return "double"
        if total_runs == 3:
            return "triple"

    # Dot ball
    if play_type == "no run" and not parsed_row["dismissal_flag"]:
        return "dot_ball"

    return "other"


def load_commentary_rows(csv_path):
    """
    Load the Kaggle commentary CSV and return a list of parsed commentary rows,
    each with an added 'event_type' field.
    """
    df = pd.read_csv(csv_path)

    if "rows" not in df.columns:
        raise ValueError("Expected a 'rows' column in the commentary CSV.")

    parsed_rows = []

    for raw_row in df["rows"].astype(str):
        parsed = parse_commentary_row(raw_row)
        parsed["event_type"] = map_commentary_to_event_type(parsed)
        parsed_rows.append(parsed)

    return parsed_rows


def build_commentary_bank(csv_path):
    """
    Build a commentary bank grouped by event type.

    Returns a dictionary like:
    {
        "boundary_four": [...],
        "wicket_caught": [...],
        ...
    }
    """
    parsed_rows = load_commentary_rows(csv_path)

    bank = {label: [] for label in EVENT_TYPES}

    for row in parsed_rows:
        label = row["event_type"]
        commentary = row["commentary"]

        if commentary:
            bank[label].append(commentary)

    return bank


if __name__ == "__main__":
    commentary_rows = load_commentary_rows("raw/train.csv")
    commentary_bank = build_commentary_bank("raw/train.csv")

    print(f"Loaded {len(commentary_rows)} commentary rows.\n")

    print("First 3 parsed rows:")
    for row in commentary_rows[:3]:
        print(row)
        print()

    print("Commentary bank counts:")
    for label, items in commentary_bank.items():
        print(f"{label}: {len(items)}")

        blank_commentary_count = sum(1 for row in commentary_rows if not row["commentary"])
    print(f"\nBlank commentary rows: {blank_commentary_count}")

    print("\nSample blank commentary rows:")
    shown = 0
    for row in commentary_rows:
        if not row["commentary"]:
            print(row["raw_text"][:300])
            print()
            shown += 1
            if shown == 5:
                break