import pandas as pd

from soccer.classify_soccer_event import SOCCER_EVENT_TYPES


def _clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_soccer_event_type(value, commentary_text=""):
    """
    Normalize commentary-side event labels into the same shared soccer label set.
    Falls back to light commentary-text checks when the source label is vague.
    """
    text = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    commentary = _clean_text(commentary_text).lower()

    if text == "goal" or "goal!" in commentary or commentary.startswith("goal"):
        return "goal"

    if text in {"save", "goalkeeper_save", "keeper_save"}:
        return "save"
    if "what a save" in commentary or "saved by" in commentary or "denies" in commentary:
        return "save"

    if text == "shot":
        return "shot"

    if text == "corner":
        return "corner"

    if text in {"free_kick", "freekick"}:
        return "free_kick"

    if text == "foul":
        return "foul"

    if text == "yellow_card":
        return "yellow_card"

    if text == "red_card":
        return "red_card"

    if text == "offside":
        return "offside"

    if text == "substitution":
        return "substitution"

    if text == "pass":
        return "pass"

    return "other"

def load_soccer_commentary_rows(csv_path):
    """
    Load the uploaded YallaShoot commentary CSV.

    Expected columns from your file:
    - id
    - match_id
    - minute
    - commentary
    - event_type
    - sentiment
    - player_mentioned
    - team
    - league
    - language
    """
    df = pd.read_csv(csv_path)

    required = {"commentary", "event_type", "minute", "player_mentioned", "team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in soccer commentary CSV: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        commentary = _clean_text(row.get("commentary"))
        event_type = _normalize_soccer_event_type(
            row.get("event_type"),
            row.get("commentary", "")
        )

        rows.append(
            {
                "sport": "soccer",
                "match_id": _clean_text(row.get("match_id")),
                "minute": row.get("minute", 0),
                "commentary": commentary,
                "event_type": event_type,
                "player": _clean_text(row.get("player_mentioned")),
                "team": _clean_text(row.get("team")),
                "league": _clean_text(row.get("league")),
                "language": _clean_text(row.get("language")),
                "sentiment": _clean_text(row.get("sentiment")),
            }
        )

    return rows


def build_soccer_commentary_bank(csv_path):
    """
    Group commentary lines by soccer event type for retrieval.
    """
    rows = load_soccer_commentary_rows(csv_path)
    bank = {label: [] for label in SOCCER_EVENT_TYPES}

    for row in rows:
        label = row["event_type"]
        commentary = row["commentary"]
        if commentary:
            bank[label].append(commentary)

    return bank


if __name__ == "__main__":
    rows = load_soccer_commentary_rows("raw_soccer/yallashoot/commentary_train.csv")
    bank = build_soccer_commentary_bank("raw_soccer/yallashoot/commentary_train.csv")

    print(f"Loaded {len(rows)} soccer commentary rows.\n")
    print("First 3 rows:")
    for row in rows[:3]:
        print(row)
        print()

    print("Bank counts:")
    for label, items in bank.items():
        print(f"{label}: {len(items)}")