"""
retriever.py
------------
Shared retrieval logic for commentary examples.

This version supports:
- event-type filtering
- in-house TF-IDF ranking within an event bucket
- sport-aware query building for both cricket and soccer
"""

from src.common.tfidf_vectorizer import TfidfVectorizerInHouse, cosine_similarity_sparse

def _join_nonempty(parts):
    """
    Join only non-empty string parts into one query string.
    """
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _build_cricket_query(event, event_type):
    """
    Build a small cricket-specific retrieval query.
    """
    return _join_nonempty(
        [
            "cricket",
            event_type,
            event.get("batter", ""),
            event.get("bowler", ""),
            str(event.get("runs_off_bat", "")),
            event.get("wicket_type", ""),
            f"over {event.get('over', '')}",
            f"ball {event.get('ball_in_over', '')}",
            event.get("batting_team", ""),
        ]
    )


def _build_soccer_query(event, event_type):
    """
    Build a small soccer-specific retrieval query.
    """
    return _join_nonempty(
        [
            "soccer",
            event_type,
            event.get("player", ""),
            event.get("team", ""),
            event.get("position", ""),
            event.get("event_type_raw", ""),
            event.get("pass_type", ""),
            event.get("pass_outcome", ""),
            event.get("shot_outcome", ""),
            event.get("shot_type", ""),
            event.get("goalkeeper_type", ""),
            event.get("goalkeeper_outcome", ""),
            event.get("foul_card", ""),
            f"minute {event.get('minute', '')}",
            f"second {event.get('second', '')}",
        ]
    )


def build_query_from_event(event, event_type):
    """
    Build a sport-aware retrieval query from event data.

    Falls back to the cricket-shaped fields if sport is missing, so older
    cricket code still works even if it does not explicitly set sport.
    """
    sport = str(event.get("sport", "")).strip().lower()

    if sport == "soccer":
        return _build_soccer_query(event, event_type)

    return _build_cricket_query(event, event_type)


def get_commentary_examples(commentary_bank, event_type, event=None, k=3):
    """
    Retrieve up to k ranked commentary examples for a given event type.

    If event is provided, rank by in-house TF-IDF similarity.
    If event is missing, just return the first k examples from that bucket.
    """
    examples = commentary_bank.get(event_type, [])

    if not examples:
        return []

    if event is None:
        return examples[:k]

    query = build_query_from_event(event, event_type)

    vectorizer = TfidfVectorizerInHouse()
    doc_vectors = vectorizer.fit_transform(examples)
    query_vector = vectorizer.transform_one(query)

    scored = []
    for example, vec in zip(examples, doc_vectors):
        score = cosine_similarity_sparse(query_vector, vec)
        scored.append((score, example))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [example for _, example in scored[:k]]


def get_commentary_examples_with_fallback(commentary_bank, event_type, event=None, k=3):
    """
    Retrieve ranked examples, falling back to 'other' if needed.
    """
    examples = get_commentary_examples(commentary_bank, event_type, event=event, k=k)

    if examples:
        return examples

    return get_commentary_examples(commentary_bank, "other", event=event, k=k)