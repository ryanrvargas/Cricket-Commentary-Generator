from generator import generate_commentary, _style_key


EVENT = {
    "batter": "Rizwan",
    "bowler": "Ali",
    "player_dismissed": "",
}


def _many_outputs(event_type, examples, count=100):
    return [
        generate_commentary(EVENT, event_type, examples).lower()
        for _ in range(count)
    ]


def test_boundary_four_pull_style_does_not_generate_drive_language():
    examples = ["Rizwan pulls this with control and gets four"]

    assert _style_key("boundary_four", examples) == "pull"

    outputs = _many_outputs("boundary_four", examples)

    assert all("pull" in line for line in outputs)
    assert all("drive" not in line for line in outputs)
    assert all("cut" not in line for line in outputs)
    assert all("flick" not in line for line in outputs)


def test_boundary_four_drive_style_does_not_generate_pull_language():
    examples = ["Rizwan drives through cover for four"]

    assert _style_key("boundary_four", examples) == "drive"

    outputs = _many_outputs("boundary_four", examples)

    assert all("drive" in line for line in outputs)
    assert all("pull" not in line for line in outputs)
    assert all("cut" not in line for line in outputs)
    assert all("flick" not in line for line in outputs)


def test_boundary_four_edge_style_does_not_generate_controlled_shot_language():
    examples = ["Rizwan gets an edge and it runs away"]

    assert _style_key("boundary_four", examples) == "edge"

    outputs = _many_outputs("boundary_four", examples)

    assert all("edge" in line for line in outputs)
    assert all("drive" not in line for line in outputs)
    assert all("pull" not in line for line in outputs)
    assert all("cut" not in line for line in outputs)


def test_boundary_six_pull_style_stays_pull_based():
    examples = ["Rizwan pulls it into the crowd for six"]

    assert _style_key("boundary_six", examples) == "pull"

    outputs = _many_outputs("boundary_six", examples)

    assert all("pull" in line for line in outputs)
    assert all("loft" not in line for line in outputs)
    assert all("straight" not in line for line in outputs)
