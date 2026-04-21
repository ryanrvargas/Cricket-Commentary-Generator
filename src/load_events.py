"""
load_events.py
--------------

This module provides utilities for loading and flattening cricket match event data from Cricsheet JSON files.

Main functionality:
- Reads a Cricsheet match JSON file.
- Flattens each delivery into a simple event dictionary for downstream processing (e.g., classification, commentary generation).
- Handles missing or blank numeric values robustly.
- Tracks both raw delivery order and legal ball number within each over, accounting for extras (wides, no-balls).

Typical usage:
    events = load_match_events('raw/1527574.json')
    # events is a list of dicts, one per delivery

Author: [Your Name]
Date: 2026-04-21
"""

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
        Load and flatten all deliveries from a Cricsheet JSON match file.

        Args:
            json_path (str or Path): Path to a Cricsheet match JSON file.

        Returns:
            list of dict: Each dict represents a single delivery (event) with fields:
                - match_id: str, unique match identifier (from filename)
                - innings: int, 1-based innings number
                - batting_team: str, name of batting team
                - over: int, over number within the innings
                - delivery_index: int, raw sequence order within the over (1-based)
                - ball_in_over: int, legal ball number (ignores wides/noballs)
                - batter: str, name of batter
                - bowler: str, name of bowler
                - non_striker: str, name of non-striker
                - runs_off_bat: int, runs scored off the bat
                - extras: int, total extras for the delivery
                - wides: int, wides bowled (0 if none)
                - noballs: int, no-balls bowled (0 if none)
                - byes: int, byes run (0 if none)
                - legbyes: int, leg byes run (0 if none)
                - wicket_type: str, type of wicket (if any, else empty)
                - player_dismissed: str, name of dismissed player (if any, else empty)

        Notes:
            - Wides and no-balls do not increment the legal ball number (ball_in_over).
            - Only the first wicket in a delivery is recorded (if multiple, only first is used).
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