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


def _example_text(retrieved_examples):
    """
    Use the first retrieved example as the style source.

    The retriever already ranks examples, so the first item should be the most
    relevant style example for the current event.
    """
    if not retrieved_examples:
        return ""
    return _normalize_example(retrieved_examples[0])


def _style_key(event_type, retrieved_examples):
    """
    Return a small style category instead of a sentence fragment.

    This prevents shaky phrasing like:
        "drives it away for four with a controlled pull"

    The generator should pick one coherent template family, not choose a base
    template and then bolt on a second shot type afterward.
    """
    example = _example_text(retrieved_examples)
    if not example:
        return "neutral"

    if event_type == "boundary_four":
        # Edge comes first because an edged four is a very different image from
        # a controlled shot like a drive, pull, cut, or flick.
        if "edge" in example or "edged" in example:
            return "edge"
        if "drive" in example or "drives" in example or "driven" in example:
            return "drive"
        if "cut" in example:
            return "cut"
        if "flick" in example or "pads" in example or "pad" in example:
            return "flick"
        if "pull" in example or "pulled" in example:
            return "pull"
        return "timing"

    if event_type == "boundary_six":
        if "pull" in example or "pulled" in example:
            return "pull"
        if "loft" in example or "lofted" in example:
            return "loft"
        if "straight" in example:
            return "straight"
        if "stands" in example or "crowd" in example:
            return "stands"
        return "power"

    if event_type == "wicket_caught":
        if "top edge" in example:
            return "top_edge"
        if "edge" in example or "edged" in example or "snick" in example:
            return "edge"
        if "holes out" in example or "holed out" in example:
            return "holes_out"
        return "safe_catch"

    return "neutral"


def _style_hint(event_type, retrieved_examples):
    """
    Use the top retrieved example to add a SMALL style hint.

    Boundary shots are intentionally excluded here. They are handled by
    _boundary_four_line and _boundary_six_line so the shot type stays coherent.
    """
    if not retrieved_examples:
        return ""

    example = _example_text(retrieved_examples)

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
        # Check top edge before edge so the more specific phrase wins.
        if "top edge" in example:
            return " off the top edge"
        if "edge" in example or "edged" in example or "snick" in example:
            return " after finding the edge"
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


def _boundary_four_line(batter, retrieved_examples):
    """
    Pick a four-run template from one coherent shot family.
    """
    style = _style_key("boundary_four", retrieved_examples)

    templates_by_style = {
        "drive": [
            f"{batter} drives it away for four.",
            f"Crisp drive from {batter}, and that runs away for four.",
            f"{batter} leans into the drive and finds the boundary."
        ],
        "cut": [
            f"{batter} cuts it away for four.",
            f"Sharp cut from {batter}, and that reaches the fence.",
            f"{batter} uses the width and cuts it for four."
        ],
        "flick": [
            f"{batter} flicks it off the pads for four.",
            f"Neatly worked off the pads by {batter}, and it races away.",
            f"{batter} times the flick well and gets four."
        ],
        "pull": [
            f"{batter} pulls it away for four.",
            f"Controlled pull from {batter}, and that reaches the boundary.",
            f"{batter} gets onto the short ball and pulls it for four."
        ],
        "edge": [
            f"{batter} gets a thick edge, and it runs away for four.",
            f"Four runs, but not exactly where {batter} intended. It flies off the edge.",
            f"An edge from {batter}, and there is no stopping it before the rope."
        ],
        "timing": [
            f"Beautifully timed by {batter}, and that is four.",
            f"Four runs. {batter} finds the boundary with good timing.",
            f"That races to the fence. Four for {batter}."
        ],
        "neutral": [
            f"Four runs. {batter} finds the boundary.",
            f"That races to the fence. Four for {batter}.",
            f"Beautifully timed by {batter}, and that is four."
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _boundary_six_line(batter, retrieved_examples):
    """
    Pick a six-run template from one coherent shot family.
    """
    style = _style_key("boundary_six", retrieved_examples)

    templates_by_style = {
        "pull": [
            f"{batter} pulls it high and clears the rope.",
            f"Powerful pull from {batter}, and that is six.",
            f"{batter} pulls the short ball away for six."
        ],
        "loft": [
            f"{batter} lofts it cleanly for six.",
            f"That is lofted beautifully by {batter}, and it clears the boundary.",
            f"{batter} lofts it high and gets all of it. Six runs."
        ],
        "straight": [
            f"{batter} sends it straight back over the bowler for six.",
            f"Straight down the ground from {batter}, and that is six.",
            f"{batter} launches it straight and clears the rope."
        ],
        "stands": [
            f"{batter} sends it deep into the stands.",
            f"That is into the crowd. Six for {batter}.",
            f"{batter} clears the rope and picks out the crowd."
        ],
        "power": [
            f"Six. {batter} sends it over the ropes.",
            f"{batter} launches it for six.",
            f"That is huge. {batter} clears the boundary."
        ],
        "neutral": [
            f"Six. {batter} sends it over the ropes.",
            f"{batter} launches it for six.",
            f"That is huge. {batter} clears the boundary."
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _wicket_caught_line(player_out, bowler, retrieved_examples):
    """
    Pick a caught-wicket template from one coherent dismissal family.
    """
    style = _style_key("wicket_caught", retrieved_examples)

    templates_by_style = {
        "top_edge": [
            f"{player_out} gets a top edge, and the catch is taken off {bowler}.",
            f"Top edge from {player_out}, and {bowler} has the wicket.",
            f"{player_out} miscues it off the top edge and is caught."
        ],
        "edge": [
            f"There is the edge. {player_out} is caught off {bowler}.",
            f"{player_out} edges it, and the catch is taken.",
            f"A nick from {player_out}, and {bowler} gets the wicket."
        ],
        "holes_out": [
            f"{player_out} holes out, and {bowler} gets the wicket.",
            f"{player_out} picks out the fielder and has to go.",
            f"Caught in the deep. {player_out} is gone off {bowler}."
        ],
        "safe_catch": [
            f"Taken. {player_out} is out caught off {bowler}.",
            f"Caught. {player_out} has to go.",
            f"{bowler} gets the breakthrough, and {player_out} is caught."
        ],
        "neutral": [
            f"Taken. {player_out} is out caught off {bowler}.",
            f"Caught. {player_out} has to go.",
            f"{bowler} gets the breakthrough, and {player_out} is caught."
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def generate_commentary(event, event_type, retrieved_examples=None, context=None):
    if retrieved_examples is None:
        retrieved_examples = []

    batter = event.get("batter", "The batter")
    bowler = event.get("bowler", "the bowler")
    player_out = event.get("player_dismissed", batter)

    # Boundary events use style-specific templates instead of appended hints.
    if event_type == "boundary_four":
        return _boundary_four_line(batter, retrieved_examples) + _context_tail(context)

    if event_type == "boundary_six":
        return _boundary_six_line(batter, retrieved_examples) + _context_tail(context)

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

    if event_type == "wicket_bowled":
        line = _choose([
            f"Bowled him. {player_out} is gone, and {bowler} strikes.",
            f"{bowler} knocks him over. {player_out} has to depart.",
            f"Cleaned up. {player_out} is bowled.",
            f"What a ball. {player_out} is out bowled."
        ])
        return _apply_style(line, hint) + _context_tail(context)

    if event_type == "wicket_caught":
        return _wicket_caught_line(player_out, bowler, retrieved_examples) + _context_tail(context)

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
