from generator import generate_commentary, _style_key


EVENT = {
    "batter": "Rizwan",
    "bowler": "Ali",
    "player_dismissed": "Rizwan",
}


def _many_outputs(event_type, examples, count=100):
    return [
        generate_commentary(EVENT, event_type, examples).lower()
        for _ in range(count)
    ]


def test_single_does_not_repeat_strike_rotation_language():
    outputs = _many_outputs("single", ["tapped away to rotate the strike"])

    assert all("keeps the strike moving to rotate the strike" not in line for line in outputs)
    assert all("to rotate the strike" not in line for line in outputs)


def test_dot_ball_does_not_append_no_scoring_chance_to_no_run_line():
    outputs = _many_outputs("dot_ball", ["defended with no scoring chance"])

    assert all("cannot score with no scoring chance" not in line for line in outputs)
    assert all("dot ball with no scoring chance" not in line for line in outputs)


def test_caught_wicket_does_not_mix_holing_out_with_safe_catch_hint():
    outputs = _many_outputs("wicket_caught", ["holes out to deep midwicket"])

    assert _style_key("wicket_caught", ["holes out to deep midwicket"]) == "holes_out"
    assert all("holes out" not in line or "catch is taken safely" not in line for line in outputs)
    assert all("as the catch is taken safely" not in line for line in outputs)


def test_boundary_four_has_no_controlled_pull_hint_phrase():
    outputs = _many_outputs("boundary_four", ["Rizwan pulls this away for four"])

    assert all("with a controlled pull" not in line for line in outputs)
    assert all("controlled pull" not in line for line in outputs)
