"""
retriever.py
------------
Retrieval logic for commentary examples.

This version supports:
- simple event-type filtering
- in-house TF-IDF ranking within an event bucket
"""

from tfidf_vectorizer import TfidfVectorizerInHouse, cosine_similarity_sparse


def build_query_from_event(event, event_type):
    """
    Build a small text query from event data for retrieval.
    """
    batter = event.get("batter", "")
    bowler = event.get("bowler", "")
    runs_off_bat = event.get("runs_off_bat", 0)
    wicket_type = event.get("wicket_type", "")
    over = event.get("over", "")

    parts = [
        event_type,
        batter,
        bowler,
        str(runs_off_bat),
        wicket_type,
        f"over {over}",
    ]

    return " ".join(str(part) for part in parts if str(part).strip())


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