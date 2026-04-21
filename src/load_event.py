import json
from pathlib import Path


def _safe_int(value):
    """
    Convert missing or blank numeric values to 0.
    """
    if value is None:
        return 0
    if value == "":
        return 0
    return int(value)


def load_match_events(json_path):
    """
    Load one Cricsheet JSON match file and flatten every delivery into a
    simple event dictionary for downstream classification and generation.

    Each flattened row includes:
    - raw sequence order within the over (delivery_index)
    - cricket-style legal ball number within the over (ball_in_over)
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    match_id = json_path.stem
    flattened_events = []

    innings_list = match_data.get("innings", [])

    for innings_index, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team", "")
        overs = innings.get("overs", [])

        for over_data in overs:
            over_number = over_data.get("over", 0)
            deliveries = over_data.get("deliveries", [])

            legal_ball_number = 0

            for delivery_index, delivery in enumerate(deliveries, start=1):
                runs = delivery.get("runs", {})
                extras_dict = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                runs_off_bat = _safe_int(runs.get("batter", 0))
                extras = _safe_int(runs.get("extras", 0))
                wides = _safe_int(extras_dict.get("wides", 0))
                noballs = _safe_int(extras_dict.get("noballs", 0))
                byes = _safe_int(extras_dict.get("byes", 0))
                legbyes = _safe_int(extras_dict.get("legbyes", 0))

                wicket_type = ""
                player_dismissed = ""

                if wickets:
                    first_wicket = wickets[0]
                    wicket_type = first_wicket.get("kind", "")
                    player_dismissed = first_wicket.get("player_out", "")

                # Wides and no-balls do not count as legal deliveries.
                if wides == 0 and noballs == 0:
                    legal_ball_number += 1

                event = {
                    "match_id": match_id,
                    "innings": innings_index,
                    "batting_team": batting_team,
                    "over": over_number,
                    "delivery_index": delivery_index,
                    "ball_in_over": legal_ball_number,
                    "batter": delivery.get("batter", ""),
                    "bowler": delivery.get("bowler", ""),
                    "non_striker": delivery.get("non_striker", ""),
                    "runs_off_bat": runs_off_bat,
                    "extras": extras,
                    "wides": wides,
                    "noballs": noballs,
                    "byes": byes,
                    "legbyes": legbyes,
                    "wicket_type": wicket_type,
                    "player_dismissed": player_dismissed,
                }

                flattened_events.append(event)

    return flattened_events