import random


def _choose(options):
    return random.choice(options)


def _context_tail(context):
    """
    Add a short live-match context phrase when available.
    """
    if not context:
        return ""

    score = f"{context['innings_score']}/{context['innings_wickets']}"

    if context.get("is_end_of_over"):
        return f" End of the over, {score}."

    if context.get("wickets_lost_on_ball") == 1:
        return f" The score is now {score}."

    if context.get("runs_on_ball", 0) >= 4:
        return f" That moves them to {score}."

    if context.get("runs_on_ball", 0) > 0:
        return f" They move to {score}."

    return f" Still {score}."


def generate_commentary(event, event_type, retrieved_examples=None, context=None):
    if retrieved_examples is None:
        retrieved_examples = []

    batter = event.get("batter", "The batter")
    bowler = event.get("bowler", "the bowler")
    player_out = event.get("player_dismissed", batter)

    if event_type == "dot_ball":
        line = _choose([
            f"Good delivery from {bowler}. {batter} cannot score.",
            f"{bowler} keeps it tight, and {batter} gets no run.",
            f"No run. {batter} is beaten for scoring options by {bowler}.",
            f"{batter} can only defend it. Dot ball."
        ])
        return line + _context_tail(context)

    if event_type == "single":
        line = _choose([
            f"{batter} works it away for a single.",
            f"Just one for {batter}.",
            f"{batter} picks up a single and keeps the strike moving.",
            f"Tapped away by {batter} for one."
        ])
        return line + _context_tail(context)

    if event_type == "double":
        line = _choose([
            f"{batter} comes back for two.",
            f"Good running. {batter} collects a couple.",
            f"That is nicely placed, and they get two.",
            f"Two runs taken by {batter}."
        ])
        return line + _context_tail(context)

    if event_type == "triple":
        line = _choose([
            f"{batter} places it well and they run three.",
            f"That is excellent placement. {batter} gets three.",
            f"They come back for three, well run by the batters."
        ])
        return line + _context_tail(context)

    if event_type == "boundary_four":
        line = _choose([
            f"Four runs. {batter} finds the boundary.",
            f"{batter} drives it away for four.",
            f"That races to the fence. Four for {batter}.",
            f"Beautifully timed by {batter}, and that is four."
        ])
        return line + _context_tail(context)

    if event_type == "boundary_six":
        line = _choose([
            f"Six. {batter} sends it over the ropes.",
            f"{batter} launches it for six.",
            f"That is huge. {batter} clears the boundary.",
            f"Straight into the stands. Six runs."
        ])
        return line + _context_tail(context)

    if event_type == "wicket_bowled":
        line = _choose([
            f"Bowled him. {player_out} is gone, and {bowler} strikes.",
            f"{bowler} knocks him over. {player_out} has to depart.",
            f"Cleaned up. {player_out} is bowled.",
            f"What a ball. {player_out} is out bowled."
        ])
        return line + _context_tail(context)

    if event_type == "wicket_caught":
        line = _choose([
            f"Taken. {player_out} is out caught off {bowler}.",
            f"{player_out} holes out, and {bowler} gets the wicket.",
            f"Caught. {player_out} has to go.",
            f"{bowler} gets the breakthrough, and {player_out} is caught."
        ])
        return line + _context_tail(context)

    if event_type == "wicket_lbw":
        line = _choose([
            f"That is out lbw. {player_out} has to go.",
            f"Huge shout, and given. {player_out} is trapped lbw.",
            f"{player_out} is out in front. LBW.",
            f"Straight enough, and {player_out} is gone lbw."
        ])
        return line + _context_tail(context)

    if event_type == "run_out":
        line = _choose([
            f"Run out. {player_out} is short of the crease.",
            f"Direct hit, and {player_out} is gone.",
            f"{player_out} cannot make the ground. Run out.",
            f"Sharp fielding brings about the run out of {player_out}."
        ])
        return line + _context_tail(context)

    if event_type == "wide":
        line = _choose([
            f"Wide ball. {bowler} sprays it too far outside.",
            f"That is called wide.",
            f"{bowler} loses the line there, and it is a wide.",
            f"Too wide to play at. Extra run conceded."
        ])
        return line + _context_tail(context)

    if event_type == "no_ball":
        line = _choose([
            f"No-ball called against {bowler}.",
            f"{bowler} has overstepped. No-ball.",
            f"That will be a no-ball.",
            f"Free run for the batting side as {bowler} is called for a no-ball."
        ])
        return line + _context_tail(context)

    if event_type == "bye_or_legbye":
        line = _choose([
            "They pick up extras, not off the bat.",
            "That will go down as extras.",
            "Not from the bat, but they still get a run.",
            "Extras added to the total."
        ])
        return line + _context_tail(context)

    return f"{batter} faces {bowler}, and the play results in {event_type}." + _context_tail(context)