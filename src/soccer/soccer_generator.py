import random


def _choose(options):
    return random.choice(options)


def _context_tail(context):
    """
    Add a small live-match tail when available.
    """
    if not context:
        return ""

    if context.get("goal_scored") and context.get("scoreline"):
        return f" The score is now {context['scoreline']}."

    return ""


def generate_soccer_commentary(event, event_type, retrieved_examples=None, context=None):
    """
    Generate one short soccer commentary line.
    """
    if retrieved_examples is None:
        retrieved_examples = []

    player = event.get("player", "The player")
    team = event.get("team", "The team")
    replacement = event.get("substitution_replacement", "a substitute")

    if event_type == "goal":
        line = _choose([
            f"Goal for {team}! {player} finds the net.",
            f"{player} scores for {team}.",
            f"It is in! {player} gives {team} the goal.",
        ])
        return line + _context_tail(context)

    if event_type == "shot":
        line = _choose([
            f"{player} gets a shot away for {team}.",
            f"An effort from {player}, but no goal.",
            f"{player} tries one for {team}.",
        ])
        return line + _context_tail(context)

    if event_type == "save":
        line = _choose([
            f"Saved by {player}.",
            f"{player} makes the stop.",
            f"A good save keeps it out.",
        ])
        return line + _context_tail(context)

    if event_type == "pass":
        line = _choose([
            f"{player} moves it on for {team}.",
            f"{player} keeps the play moving for {team}.",
            f"A tidy pass from {player}.",
        ])
        return line + _context_tail(context)

    if event_type == "corner":
        line = _choose([
            f"Corner to {team}.",
            f"{team} win a corner.",
            f"The pressure stays on with a corner for {team}.",
        ])
        return line + _context_tail(context)

    if event_type == "free_kick":
        line = _choose([
            f"Free kick for {team}.",
            f"{team} have a set-piece chance.",
            f"A free-kick opportunity here for {team}.",
        ])
        return line + _context_tail(context)

    if event_type == "foul":
        line = _choose([
            f"Foul by {player}.",
            f"The referee gives the foul.",
            f"Play is stopped for a foul.",
        ])
        return line + _context_tail(context)

    if event_type == "yellow_card":
        line = _choose([
            f"Yellow card shown to {player}.",
            f"{player} goes into the book.",
            f"The referee reaches for the yellow card.",
        ])
        return line + _context_tail(context)

    if event_type == "red_card":
        line = _choose([
            f"Red card for {player}.",
            f"{player} is sent off.",
            f"The referee shows red.",
        ])
        return line + _context_tail(context)

    if event_type == "offside":
        line = _choose([
            f"Offside against {team}.",
            f"The flag is up for offside.",
            f"Play is stopped for offside.",
        ])
        return line + _context_tail(context)

    if event_type == "substitution":
        line = _choose([
            f"Substitution for {team}, with {replacement} coming on.",
            f"{team} make a change.",
            f"A substitution here for {team}.",
        ])
        return line + _context_tail(context)

    line = _choose([
        f"{team} keep the move alive.",
        f"Play continues for {team}.",
        "No major change in the phase of play.",
    ])
    return line + _context_tail(context)