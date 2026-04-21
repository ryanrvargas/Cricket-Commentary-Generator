import random


def get_commentary_examples(commentary_bank, event_type, k=3, seed=42):
    """
    Return up to k commentary examples for a given event type.

    If there are fewer than k examples, return all of them.
    If the event type does not exist, return an empty list.
    """
    examples = commentary_bank.get(event_type, [])

    if not examples:
        return []

    if len(examples) <= k:
        return examples

    rng = random.Random(seed)
    return rng.sample(examples, k)


def get_commentary_examples_with_fallback(commentary_bank, event_type, k=3, seed=42):
    """
    Return up to k commentary examples for a given event type.

    If that bucket is empty, fall back to 'other'.
    """
    examples = get_commentary_examples(commentary_bank, event_type, k=k, seed=seed)

    if examples:
        return examples

    return get_commentary_examples(commentary_bank, "other", k=k, seed=seed)