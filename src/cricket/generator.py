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


def _has_any(text, keywords):
    """
    Return True if any keyword appears in the normalized example text.
    """
    return any(keyword in text for keyword in keywords)


def _style_key(event_type, retrieved_examples):
    """
    Return a small style category instead of a sentence fragment.

    This keeps retrieval useful without bolting risky fragments onto templates.
    Each event type chooses one coherent template family.
    """
    example = _example_text(retrieved_examples)
    if not example:
        return "neutral"

    if event_type == "boundary_four":
        # Edge comes first because an edged four is a very different image from
        # a controlled shot like a drive, pull, cut, or flick.
        if _has_any(example, ["edge", "edged", "outside edge", "inside edge"]):
            return "edge"
        if _has_any(example, ["drive", "drives", "driven", "cover"]):
            return "drive"
        if _has_any(example, ["cut", "cuts", "width"]):
            return "cut"
        if _has_any(example, ["flick", "flicks", "pads", "pad", "worked"]):
            return "flick"
        if _has_any(example, ["pull", "pulled", "short ball"]):
            return "pull"
        return "timing"

    if event_type == "boundary_six":
        if _has_any(example, ["pull", "pulled", "short ball"]):
            return "pull"
        if _has_any(example, ["loft", "lofted", "high"]):
            return "loft"
        if _has_any(example, ["straight", "down the ground"]):
            return "straight"
        if _has_any(example, ["stands", "crowd"]):
            return "stands"
        return "power"

    if event_type == "single":
        if _has_any(example, ["leg side", "square leg", "fine leg", "pads", "pad"]):
            return "leg_side"
        if _has_any(example, ["mid off", "mid-off", "mid on", "mid-on"]):
            return "mid_off"
        if _has_any(example, ["soft hands", "softly"]):
            return "soft_hands"
        if _has_any(example, ["tap", "tapped", "nudge", "nudged"]):
            return "tap"
        return "rotation"

    if event_type == "double":
        if _has_any(example, ["gap", "placed", "placement"]):
            return "placement"
        return "running"

    if event_type == "triple":
        if _has_any(example, ["gap", "deep", "outfield"]):
            return "gap"
        return "running"

    if event_type == "dot_ball":
        if _has_any(example, ["defend", "defended", "defence", "defense"]):
            return "defence"
        if _has_any(example, ["leave", "left alone"]):
            return "leave"
        if _has_any(example, ["beaten", "beats", "beat"]):
            return "beaten"
        return "tight"

    if event_type == "wicket_caught":
        if "top edge" in example:
            return "top_edge"
        if _has_any(example, ["edge", "edged", "snick", "nick"]):
            return "edge"
        if _has_any(example, ["holes out", "holed out", "deep"]):
            return "holes_out"
        return "safe_catch"

    if event_type == "wicket_bowled":
        if "gate" in example:
            return "gate"
        if _has_any(example, ["cleaned up", "knocked over"]):
            return "cleaned_up"
        return "stumps"

    if event_type == "wicket_lbw":
        if _has_any(example, ["trapped", "plumb"]):
            return "trapped"
        return "front"

    if event_type == "run_out":
        if "direct hit" in example:
            return "direct_hit"
        return "sharp_fielding"

    if event_type == "wide":
        if "bouncer" in example:
            return "bouncer"
        if _has_any(example, ["outside off", "wide outside"]):
            return "outside_off"
        return "line"

    if event_type == "bye_or_legbye":
        if _has_any(example, ["thigh", "pad", "pads", "leg bye"]):
            return "pads"
        return "extras"

    return "neutral"


def _boundary_four_line(batter, retrieved_examples):
    """
    Pick a four-run template from one coherent shot family.
    """
    style = _style_key("boundary_four", retrieved_examples)

    templates_by_style = {
        "drive": [
            f"{batter} drives it away for four.",
            f"Crisp drive from {batter}, and that runs away for four.",
            f"{batter} leans into the drive and finds the boundary.",
        ],
        "cut": [
            f"{batter} cuts it away for four.",
            f"Sharp cut from {batter}, and that reaches the fence.",
            f"{batter} uses the width and cuts it for four.",
        ],
        "flick": [
            f"{batter} flicks it off the pads for four.",
            f"Neatly worked off the pads by {batter}, and it races away.",
            f"{batter} times the flick well and gets four.",
        ],
        "pull": [
            f"{batter} pulls it away for four.",
            f"{batter} pulls it cleanly, and that reaches the boundary.",
            f"{batter} gets onto the short ball and pulls it for four.",
        ],
        "edge": [
            f"{batter} gets a thick edge, and it runs away for four.",
            f"Four runs, but not exactly where {batter} intended. It flies off the edge.",
            f"An edge from {batter}, and there is no stopping it before the rope.",
        ],
        "timing": [
            f"Beautifully timed by {batter}, and that is four.",
            f"Four runs. {batter} finds the boundary with good timing.",
            f"That races to the fence. Four for {batter}.",
        ],
        "neutral": [
            f"Four runs. {batter} finds the boundary.",
            f"That races to the fence. Four for {batter}.",
            f"Beautifully timed by {batter}, and that is four.",
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
            f"{batter} pulls the short ball away for six.",
        ],
        "loft": [
            f"{batter} lofts it cleanly for six.",
            f"That is lofted beautifully by {batter}, and it clears the boundary.",
            f"{batter} lofts it high and gets all of it. Six runs.",
        ],
        "straight": [
            f"{batter} sends it straight back over the bowler for six.",
            f"Straight down the ground from {batter}, and that is six.",
            f"{batter} launches it straight and clears the rope.",
        ],
        "stands": [
            f"{batter} sends it deep into the stands.",
            f"That is into the crowd. Six for {batter}.",
            f"{batter} clears the rope and picks out the crowd.",
        ],
        "power": [
            f"Six. {batter} sends it over the ropes.",
            f"{batter} launches it for six.",
            f"That is huge. {batter} clears the boundary.",
        ],
        "neutral": [
            f"Six. {batter} sends it over the ropes.",
            f"{batter} launches it for six.",
            f"That is huge. {batter} clears the boundary.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _single_line(batter, retrieved_examples):
    """
    Pick a single-run template without appending redundant strike-rotation hints.
    """
    style = _style_key("single", retrieved_examples)

    templates_by_style = {
        "leg_side": [
            f"{batter} works it into the leg side for one.",
            f"One run for {batter}, clipped off the pads.",
            f"{batter} turns it around the corner for a single.",
        ],
        "mid_off": [
            f"{batter} pushes it in front of mid-off and takes one.",
            f"A gentle push from {batter}, and they get a single.",
            f"{batter} drops it near mid-off for one.",
        ],
        "soft_hands": [
            f"Soft hands from {batter}, and they take a single.",
            f"{batter} plays it softly and gets one.",
            f"A controlled touch from {batter} brings a single.",
        ],
        "tap": [
            f"{batter} taps it away for one.",
            f"Tapped into the gap by {batter} for a single.",
            f"{batter} nudges it away and takes one.",
        ],
        "rotation": [
            f"{batter} works it away for a single.",
            f"Just one for {batter}.",
            f"{batter} picks up a single.",
        ],
        "neutral": [
            f"{batter} works it away for a single.",
            f"Just one for {batter}.",
            f"{batter} picks up a single.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _double_line(batter, retrieved_examples):
    """
    Pick a two-run template from placement or running style.
    """
    style = _style_key("double", retrieved_examples)

    templates_by_style = {
        "placement": [
            f"{batter} places it into the gap and comes back for two.",
            f"Good placement from {batter}, and they pick up two.",
            f"That is nicely placed by {batter} for a couple.",
        ],
        "running": [
            f"{batter} comes back for two.",
            f"Good running from the batters, and {batter} gets a couple.",
            f"Two runs taken by {batter}.",
        ],
        "neutral": [
            f"{batter} comes back for two.",
            f"Good running from the batters, and {batter} gets a couple.",
            f"Two runs taken by {batter}.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _triple_line(batter, retrieved_examples):
    """
    Pick a three-run template from gap or running style.
    """
    style = _style_key("triple", retrieved_examples)

    templates_by_style = {
        "gap": [
            f"{batter} finds a big gap, and they run three.",
            f"That is placed deep into the outfield, and {batter} gets three.",
            f"Excellent placement from {batter}, and they come back for three.",
        ],
        "running": [
            f"They come back for three, well run by the batters.",
            f"{batter} gets three after hard running.",
            f"Three runs taken, and {batter} makes them work for it.",
        ],
        "neutral": [
            f"They come back for three, well run by the batters.",
            f"{batter} gets three after hard running.",
            f"Three runs taken, and {batter} makes them work for it.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _dot_ball_line(batter, bowler, retrieved_examples):
    """
    Pick a dot-ball template instead of stapling on a generic hint.
    """
    style = _style_key("dot_ball", retrieved_examples)

    templates_by_style = {
        "defence": [
            f"{batter} gets behind it and defends. No run.",
            f"Solid defence from {batter}. Dot ball.",
            f"{batter} meets {bowler} with a firm defence.",
        ],
        "leave": [
            f"{batter} leaves it alone. No run.",
            f"Left alone by {batter}, and it is another dot.",
            f"{batter} shoulders arms and lets it go through.",
        ],
        "beaten": [
            f"{bowler} beats {batter}. No run.",
            f"Play and miss from {batter}. Dot ball.",
            f"{batter} is beaten by {bowler}, and there is no run.",
        ],
        "tight": [
            f"Good delivery from {bowler}. No run.",
            f"{bowler} keeps it tight, and {batter} cannot score.",
            f"Dot ball. {batter} finds no scoring option.",
        ],
        "neutral": [
            f"Good delivery from {bowler}. No run.",
            f"{bowler} keeps it tight, and {batter} cannot score.",
            f"Dot ball. {batter} finds no scoring option.",
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
            f"{player_out} miscues it off the top edge and is caught.",
        ],
        "edge": [
            f"There is the edge. {player_out} is caught off {bowler}.",
            f"{player_out} edges it, and the catch is taken.",
            f"A nick from {player_out}, and {bowler} gets the wicket.",
        ],
        "holes_out": [
            f"{player_out} holes out, and {bowler} gets the wicket.",
            f"{player_out} picks out the fielder and has to go.",
            f"Caught in the deep. {player_out} is gone off {bowler}.",
        ],
        "safe_catch": [
            f"Taken. {player_out} is out caught off {bowler}.",
            f"Caught. {player_out} has to go.",
            f"{bowler} gets the breakthrough, and {player_out} is caught.",
        ],
        "neutral": [
            f"Taken. {player_out} is out caught off {bowler}.",
            f"Caught. {player_out} has to go.",
            f"{bowler} gets the breakthrough, and {player_out} is caught.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _wicket_bowled_line(player_out, bowler, retrieved_examples):
    """
    Pick a bowled-wicket template without repeating the same dismissal wording.
    """
    style = _style_key("wicket_bowled", retrieved_examples)

    templates_by_style = {
        "gate": [
            f"{bowler} goes through the gate. {player_out} is bowled.",
            f"Bowled through the gate. {player_out} has to depart.",
            f"{player_out} is beaten through the gate, and {bowler} strikes.",
        ],
        "cleaned_up": [
            f"Cleaned up. {player_out} is bowled.",
            f"{bowler} knocks over {player_out}.",
            f"{player_out} is cleaned up, and {bowler} has another.",
        ],
        "stumps": [
            f"Bowled. {player_out} is gone, and {bowler} strikes.",
            f"The stumps are hit, and {player_out} has to go.",
            f"What a ball from {bowler}. {player_out} is bowled.",
        ],
        "neutral": [
            f"Bowled. {player_out} is gone, and {bowler} strikes.",
            f"The stumps are hit, and {player_out} has to go.",
            f"What a ball from {bowler}. {player_out} is bowled.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _wicket_lbw_line(player_out, retrieved_examples):
    """
    Pick an LBW template from one coherent dismissal family.
    """
    style = _style_key("wicket_lbw", retrieved_examples)

    templates_by_style = {
        "trapped": [
            f"{player_out} is trapped in front. LBW.",
            f"Huge shout, and given. {player_out} is trapped lbw.",
            f"That is plumb. {player_out} has to go.",
        ],
        "front": [
            f"That is out lbw. {player_out} has to go.",
            f"Straight enough, and {player_out} is gone lbw.",
            f"The umpire raises the finger. {player_out} is lbw.",
        ],
        "neutral": [
            f"That is out lbw. {player_out} has to go.",
            f"Straight enough, and {player_out} is gone lbw.",
            f"The umpire raises the finger. {player_out} is lbw.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _run_out_line(player_out, retrieved_examples):
    """
    Pick a run-out template without duplicating direct-hit language.
    """
    style = _style_key("run_out", retrieved_examples)

    templates_by_style = {
        "direct_hit": [
            f"Direct hit, and {player_out} is gone.",
            f"{player_out} is short after a direct hit.",
            f"The direct hit does it. {player_out} is run out.",
        ],
        "sharp_fielding": [
            f"Run out. {player_out} is short of the crease.",
            f"{player_out} cannot make the ground. Run out.",
            f"Sharp fielding brings about the run out of {player_out}.",
        ],
        "neutral": [
            f"Run out. {player_out} is short of the crease.",
            f"{player_out} cannot make the ground. Run out.",
            f"Sharp fielding brings about the run out of {player_out}.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _wide_line(bowler, retrieved_examples):
    """
    Pick a wide-ball template from the retrieved style.
    """
    style = _style_key("wide", retrieved_examples)

    templates_by_style = {
        "bouncer": [
            f"That bouncer is called wide.",
            f"Too high from {bowler}, and it is signalled wide.",
            f"The short ball is too high. Wide called.",
        ],
        "outside_off": [
            f"That is too far outside off. Wide ball.",
            f"{bowler} misses the line outside off, and it is wide.",
            f"Too wide outside off for the batter to reach.",
        ],
        "line": [
            f"Wide ball. {bowler} loses the line.",
            f"That is called wide.",
            f"Too wide to play at. Extra run conceded.",
        ],
        "neutral": [
            f"Wide ball. {bowler} loses the line.",
            f"That is called wide.",
            f"Too wide to play at. Extra run conceded.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def _bye_or_legbye_line(retrieved_examples):
    """
    Pick an extras template for byes or leg byes.
    """
    style = _style_key("bye_or_legbye", retrieved_examples)

    templates_by_style = {
        "pads": [
            "They pick up extras off the pads.",
            "Leg byes added to the total.",
            "Not off the bat, but they still get runs from the pads.",
        ],
        "extras": [
            "They pick up extras, not off the bat.",
            "That will go down as extras.",
            "Not from the bat, but they still get a run.",
        ],
        "neutral": [
            "They pick up extras, not off the bat.",
            "That will go down as extras.",
            "Not from the bat, but they still get a run.",
        ],
    }

    return _choose(templates_by_style.get(style, templates_by_style["neutral"]))


def generate_commentary(event, event_type, retrieved_examples=None, context=None):
    if retrieved_examples is None:
        retrieved_examples = []

    batter = event.get("batter", "The batter")
    bowler = event.get("bowler", "the bowler")
    player_out = event.get("player_dismissed", batter)

    if event_type == "boundary_four":
        return _boundary_four_line(batter, retrieved_examples) + _context_tail(context)

    if event_type == "boundary_six":
        return _boundary_six_line(batter, retrieved_examples) + _context_tail(context)

    if event_type == "dot_ball":
        return _dot_ball_line(batter, bowler, retrieved_examples) + _context_tail(context)

    if event_type == "single":
        return _single_line(batter, retrieved_examples) + _context_tail(context)

    if event_type == "double":
        return _double_line(batter, retrieved_examples) + _context_tail(context)

    if event_type == "triple":
        return _triple_line(batter, retrieved_examples) + _context_tail(context)

    if event_type == "wicket_bowled":
        return _wicket_bowled_line(player_out, bowler, retrieved_examples) + _context_tail(context)

    if event_type == "wicket_caught":
        return _wicket_caught_line(player_out, bowler, retrieved_examples) + _context_tail(context)

    if event_type == "wicket_lbw":
        return _wicket_lbw_line(player_out, retrieved_examples) + _context_tail(context)

    if event_type == "run_out":
        return _run_out_line(player_out, retrieved_examples) + _context_tail(context)

    if event_type == "wide":
        return _wide_line(bowler, retrieved_examples) + _context_tail(context)

    if event_type == "no_ball":
        return _choose([
            f"No-ball called against {bowler}.",
            f"{bowler} has overstepped. No-ball.",
            f"That will be a no-ball.",
            f"Free run for the batting side as {bowler} is called for a no-ball.",
        ]) + _context_tail(context)

    if event_type == "bye_or_legbye":
        return _bye_or_legbye_line(retrieved_examples) + _context_tail(context)

    return f"{batter} faces {bowler}, and the play results in {event_type}." + _context_tail(context)
