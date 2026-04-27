import random
import re


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


def _normalize_example(text):
    """
    Lowercase and simplify a retrieved example so we can look for style words.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _style_hint(event_type, retrieved_examples):
    """
    Use the top retrieved example to add a SMALL style hint.
    This should influence wording, not facts.
    """
    if not retrieved_examples:
        return ""

    example = _normalize_example(retrieved_examples[0])

    if event_type == "boundary_four":
        if "drive" in example:
            return " with a crisp drive"
        if "cut" in example:
            return " with a neat cut shot"
        if "flick" in example or "pads" in example:
            return " with a flick off the pads"
        if "pull" in example:
            return " with a controlled pull"
        if "edge" in example:
            return " off a thick edge"
        return " with good timing"

    if event_type == "boundary_six":
        if "pull" in example:
            return " with a powerful pull"
        if "loft" in example:
            return " with a lofted hit"
        if "straight" in example:
            return " straight down the ground"
        if "stands" in example or "crowd" in example:
            return " deep into the stands"
        return " with plenty of power"

    if event_type == "single":
        if "leg side" in example:
            return " into the leg side"
        if "mid off" in example or "mid-off" in example:
            return " in front of mid-off"
        if "soft hands" in example:
            return " with soft hands"
        if "tap" in example or "tapped" in example:
            return " with a gentle tap"
        return " to rotate the strike"

    if event_type == "double":
        if "gap" in example:
            return " into the gap"
        if "placed" in example:
            return " with good placement"
        return " with quick running"

    if event_type == "triple":
        if "gap" in example:
            return " into a big gap"
        return " with excellent running"

    if event_type == "dot_ball":
        if "defend" in example or "defended" in example:
            return " after a solid defence"
        if "leave" in example or "left alone" in example:
            return " as it is left alone"
        if "beaten" in example:
            return " as the batter is beaten"
        return " with no scoring chance"

    if event_type == "wicket_caught":
        if "edge" in example or "edged" in example or "snick" in example:
            return " after finding the edge"
        if "top edge" in example:
            return " off the top edge"
        if "holes out" in example:
            return " after holing out"
        return " as the catch is taken safely"

    if event_type == "wicket_bowled":
        if "gate" in example:
            return " through the gate"
        if "cleaned up" in example:
            return " after being cleaned up"
        return " as the stumps are hit"

    if event_type == "wicket_lbw":
        if "trapped" in example:
            return " trapped in front"
        return " right in front of the stumps"

    if event_type == "run_out":
        if "direct hit" in example:
            return " with a direct hit"
        return " after sharp fielding"

    if event_type == "wide":
        if "bouncer" in example:
            return " with the bouncer going too high or wide"
        if "outside off" in example:
            return " far outside off"
        return " down the wrong line"

    if event_type == "no_ball":
        return " after overstepping"

    if event_type == "bye_or_legbye":
        if "thigh" in example or "pad" in example:
            return " off the pads"
        return " as extras"

    return ""


def _apply_style(line, hint):
    """
    Add a style hint cleanly to the end of a template sentence.
    """
    if not hint:
        return line

    line = line.rstrip(".!?")
    return f"{line}{hint}."


def generate_commentary(event, event_type, retrieved_examples=None, context=None):
    if retrieved_examples is None:
        retrieved_examples = []

    batter = event.get("batter", "The batter")
    bowler = event.get("bowler", "the bowler")
    player_out = event.get("player_dismissed", batter)

    hint = _style_hint(event_type, retrieved_examples)

    if event_type == "dot_ball":
        line = _choose([
            f"Good delivery from {bowler}. {batter} cannot score.",
            f"{bowler} keeps it tight, and {batter} gets no run.",
            f"No run. {batter} is beaten for scoring options by {bowler}.",
            f"{batter} can only defend it. Dot ball."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "single":
        line = _choose([
            f"{batter} works it away for a single.",
            f"Just one for {batter}.",
            f"{batter} picks up a single and keeps the strike moving.",
            f"Tapped away by {batter} for one."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "double":
        line = _choose([
            f"{batter} comes back for two.",
            f"Good running. {batter} collects a couple.",
            f"That is nicely placed, and they get two.",
            f"Two runs taken by {batter}."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "triple":
        line = _choose([
            f"{batter} places it well and they run three.",
            f"That is excellent placement. {batter} gets three.",
            f"They come back for three, well run by the batters."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "boundary_four":
        line = _choose([
            f"Four runs. {batter} finds the boundary.",
            f"{batter} drives it away for four.",
            f"That races to the fence. Four for {batter}.",
            f"Beautifully timed by {batter}, and that is four."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "boundary_six":
        line = _choose([
            f"Six. {batter} sends it over the ropes.",
            f"{batter} launches it for six.",
            f"That is huge. {batter} clears the boundary.",
            f"Straight into the stands. Six runs."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "wicket_bowled":
        line = _choose([
            f"Bowled him. {player_out} is gone, and {bowler} strikes.",
            f"{bowler} knocks him over. {player_out} has to depart.",
            f"Cleaned up. {player_out} is bowled.",
            f"What a ball. {player_out} is out bowled."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "wicket_caught":
        line = _choose([
            f"Taken. {player_out} is out caught off {bowler}.",
            f"{player_out} holes out, and {bowler} gets the wicket.",
            f"Caught. {player_out} has to go.",
            f"{bowler} gets the breakthrough, and {player_out} is caught."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "wicket_lbw":
        line = _choose([
            f"That is out lbw. {player_out} has to go.",
            f"Huge shout, and given. {player_out} is trapped lbw.",
            f"{player_out} is out in front. LBW.",
            f"Straight enough, and {player_out} is gone lbw."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "run_out":
        line = _choose([
            f"Run out. {player_out} is short of the crease.",
            f"Direct hit, and {player_out} is gone.",
            f"{player_out} cannot make the ground. Run out.",
            f"Sharp fielding brings about the run out of {player_out}."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "wide":
        line = _choose([
            f"Wide ball. {bowler} sprays it too far outside.",
            f"That is called wide.",
            f"{bowler} loses the line there, and it is a wide.",
            f"Too wide to play at. Extra run conceded."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "no_ball":
        line = _choose([
            f"No-ball called against {bowler}.",
            f"{bowler} has overstepped. No-ball.",
            f"That will be a no-ball.",
            f"Free run for the batting side as {bowler} is called for a no-ball."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "bye_or_legbye":
        line = _choose([
            "They pick up extras, not off the bat.",
            "That will go down as extras.",
            "Not from the bat, but they still get a run.",
            "Extras added to the total."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    return _apply_style(
        f"{batter} faces {bowler}, and the play results in {event_type}.",
        hint
    ) + _context_tail(context)