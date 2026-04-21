import random


def _pick(examples, fallback=""):
    """
    Pick one example phrase if available, otherwise return the fallback.
    """
    if not examples:
        return fallback
    return random.choice(examples)


def generate_commentary(event, event_type, retrieved_examples=None):
    """
    Generate one short commentary line for a single cricket event.

    This first version is template-based. Retrieved examples are accepted
    for future use, but templates stay in control so the output remains
    factually grounded in the event row.
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