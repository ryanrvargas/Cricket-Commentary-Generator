from load_events import load_match_events
from classify_event import classify_event
from load_commentary import build_commentary_bank
from retriever import get_commentary_examples
from generator import generate_commentary


def build_event_context(event, runs_on_ball, wickets_lost_on_ball, innings_score, innings_wickets):
    """
    Build a small context dictionary for one delivery.
    """
    return {
        "innings": event["innings"],
        "over": event["over"],
        "ball_in_over": event["ball_in_over"],
        "runs_on_ball": runs_on_ball,
        "wickets_lost_on_ball": wickets_lost_on_ball,
        "innings_score": innings_score,
        "innings_wickets": innings_wickets,
        "is_end_of_over": event["ball_in_over"] == 6,
    }


def run_match_demo(match_path, commentary_csv_path, max_events=None):
    """
    Run a sequential commentary demo for one match file and track live score context.
    """
    events = load_match_events(match_path)
    commentary_bank = build_commentary_bank(commentary_csv_path)

    if max_events is not None:
        events = events[:max_events]

    current_innings = None
    innings_score = 0
    innings_wickets = 0

    for event in events:
        # Reset scoreboard when innings changes
        if current_innings != event["innings"]:
            current_innings = event["innings"]
            innings_score = 0
            innings_wickets = 0
            print(f"\n=== Start of Innings {current_innings} ===")

        event_type = classify_event(event)
        examples = get_commentary_examples(commentary_bank, event_type, k=3)
        commentary = generate_commentary(event, event_type, examples, context)

        runs_on_ball = event["runs_off_bat"] + event["extras"]
        wickets_lost_on_ball = 1 if event["wicket_type"] else 0

        innings_score += runs_on_ball
        innings_wickets += wickets_lost_on_ball

        context = build_event_context(
            event,
            runs_on_ball,
            wickets_lost_on_ball,
            innings_score,
            innings_wickets,
        )

        print(
            f"Innings {event['innings']} | "
            f"{event['over']}.{event['ball_in_over']} | "
            f"{event['batter']} vs {event['bowler']} -> {event_type}"
        )
        print(f"Score after ball: {context['innings_score']}/{context['innings_wickets']}")
        print(f"Runs on ball: {context['runs_on_ball']}")
        print(f"Commentary: {commentary}")

        if context["is_end_of_over"]:
            print(f"End of over {event['over']}: {context['innings_score']}/{context['innings_wickets']}")

        print("-" * 70)


if __name__ == "__main__":
    run_match_demo("raw/1527574.json", "raw/train.csv", max_events=24)