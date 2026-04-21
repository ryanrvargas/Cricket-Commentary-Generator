"""
generator.py
------------
This module generates short, template-based cricket commentary lines for individual events.
It provides the main function for producing a commentary string given a structured event dictionary
and a fixed event type label. The design allows for future integration of retrieved example commentary
lines, but currently prioritizes template-based output for factual accuracy.

Functions:
    - generate_commentary(event, event_type, retrieved_examples=None):
        Generates a commentary line for a given event and event type.
    - _pick(examples, fallback=""): Utility to pick a random example or fallback.

Typical usage:
    commentary = generate_commentary(event, event_type, examples)
"""
import random


def _pick(examples, fallback=""):
    """
    Pick one example phrase from a list, or return the fallback if the list is empty.

    Args:
        examples (list): List of example strings to choose from.
        fallback (str): String to return if examples is empty.

    Returns:
        str: A randomly chosen example, or the fallback string.
    """
    if not examples:
        return fallback
    return random.choice(examples)


def generate_commentary(event, event_type, retrieved_examples=None):
    """
    Generate a short commentary line for a single cricket event.

    This function uses a set of templates to produce a commentary line based on the event type
    and event details. Optionally, retrieved example commentary lines can be provided for future
    use, but templates are always used to ensure the output is grounded in the event data.

    Args:
        event (dict): Dictionary containing event details (e.g., batter, bowler, player_dismissed).
        event_type (str): The fixed event label (e.g., 'boundary_four', 'wicket_bowled').
        retrieved_examples (list, optional): List of example commentary lines for the event type.

    Returns:
        str: A generated commentary line for the event.
    """
    if retrieved_examples is None:
        retrieved_examples = []

    batter = event.get("batter", "The batter")
    bowler = event.get("bowler", "the bowler")
    player_out = event.get("player_dismissed", batter)

    if event_type == "dot_ball":
        return f"Good delivery from {bowler}. {batter} cannot score."

    if event_type == "single":
        return f"{batter} works it away for a single."

    if event_type == "double":
        return f"{batter} comes back for two."

    if event_type == "triple":
        return f"{batter} places it well and they run three."

    if event_type == "boundary_four":
        return f"Four runs. {batter} finds the boundary."

    if event_type == "boundary_six":
        return f"Six. {batter} sends it over the ropes."

    if event_type == "wicket_bowled":
        return f"Bowled him. {player_out} is gone, and {bowler} strikes."

    if event_type == "wicket_caught":
        return f"Taken. {player_out} is out caught off {bowler}."

    if event_type == "wicket_lbw":
        return f"That is out lbw. {player_out} has to go."

    if event_type == "run_out":
        return f"Run out. {player_out} is short of the crease."

    if event_type == "wide":
        return f"Wide ball. {bowler} sprays it too far outside."

    if event_type == "no_ball":
        return f"No-ball called against {bowler}."

    if event_type == "bye_or_legbye":
        return "They pick up extras, not off the bat."

    return f"{batter} faces {bowler}, and the play results in {event_type}."