
"""
load_commentary.py
------------------
This module loads, parses, and processes cricket commentary data from the Kaggle CSV dataset.
It provides utilities to:
    - Parse the single 'rows' column in the CSV into structured fields.
    - Map each commentary line into a fixed set of event labels (see event_types.py).
    - Build a commentary bank for retrieval, grouped by event type.

Typical pipeline usage:
    1. Use load_commentary_rows(csv_path) to parse the CSV into structured rows with event labels.
    2. Use build_commentary_bank(csv_path) to group commentary lines by event type for retrieval.

Key functions:
    - parse_commentary_row: Parses a raw commentary row string into a structured dictionary.
    - map_commentary_to_event_type: Maps a parsed row to a fixed event label.
    - load_commentary_rows: Loads and parses the CSV, returning rows with event labels.
    - build_commentary_bank: Builds a dictionary of commentary examples grouped by event type.

Example event type labels: 'boundary_four', 'wicket_bowled', 'single', 'wide', etc.
"""

import re
import pandas as pd
import html

from cricket.event_types import EVENT_TYPES


def _clean_text(text):
    """
    Normalize whitespace, strip HTML tags/entities, and return a clean string.

    Args:
        text (str or None): The text to clean.

    Returns:
        str: Cleaned text with HTML removed and whitespace normalized.
    """
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def _extract(pattern, text, default=""):
    """
    Extract the first regex group from text. If no match is found, return the default value.

    Args:
        pattern (str): Regex pattern with one group to extract.
        text (str): Text to search.
        default (str): Value to return if no match is found.

    Returns:
        str: Extracted and cleaned group, or default if not found.
    """
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _clean_text(match.group(1))
    return default


def _to_int(value, default=0):
    """
    Safely convert a parsed string value to int.

    Args:
        value: Value to convert (str, float, int, etc.).
        default (int): Value to return if conversion fails.

    Returns:
        int: Converted integer, or default if conversion fails.
    """
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def parse_commentary_row(raw_row):
    """
    Parse one Kaggle commentary row from the single 'rows' column into a structured dictionary.

    The dataset stores metadata and commentary in one long text string:
        <start_of_table> ... <end_of_table> commentary ...

    Args:
        raw_row (str): The raw text from the 'rows' column.

    Returns:
        dict: Parsed fields including play_type, teams, runs, wickets, and commentary.
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

    Args:
        parsed_row (dict): Parsed commentary row from parse_commentary_row.

    Returns:
        str: Event type label (e.g., 'boundary_four', 'wicket_bowled', etc.).
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
    Load the Kaggle commentary CSV and return a list of parsed commentary rows, each with an added 'event_type' field.

    Args:
        csv_path (str): Path to the Kaggle commentary CSV file.

    Returns:
        list of dict: Each dict is a parsed row with an 'event_type' key.
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

    Args:
        csv_path (str): Path to the Kaggle commentary CSV file.

    Returns:
        dict: Dictionary mapping event type labels to lists of commentary strings.
            Example:
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