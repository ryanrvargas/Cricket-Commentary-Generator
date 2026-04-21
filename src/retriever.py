"""
retriever.py
------------

This module provides functions to retrieve example commentary lines for cricket events from a commentary bank.
It supports random sampling of examples for a given event type, with fallback logic if no examples are found.

Functions:
- get_commentary_examples: Retrieve up to k random examples for a specific event type.
- get_commentary_examples_with_fallback: Retrieve up to k examples for an event type, falling back to 'other' if none exist.

Typical usage:
    examples = get_commentary_examples(commentary_bank, 'bowled', k=3)
    fallback_examples = get_commentary_examples_with_fallback(commentary_bank, 'run_out', k=2)
"""

import random


def get_commentary_examples(commentary_bank, event_type, k=3, seed=42):
    """
    Retrieve up to k random commentary examples for a given event type.

    Args:
        commentary_bank (dict): Dictionary mapping event types (str) to lists of commentary strings.
        event_type (str): The event type to retrieve examples for (e.g., 'bowled', 'run_out').
        k (int, optional): Maximum number of examples to return. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list of str: Up to k commentary examples for the event type. If fewer than k exist, returns all. If event type is not found, returns an empty list.

    Example:
        >>> get_commentary_examples({'bowled': ['Bowled him!', 'Clean bowled!']}, 'bowled', k=1)
        ['Bowled him!']
    """
    examples = commentary_bank.get(event_type, [])

    if not examples:
        return []  # No examples for this event type

    if len(examples) <= k:
        return examples  # Return all if not enough to sample

    rng = random.Random(seed)
    return rng.sample(examples, k)  # Randomly sample k examples


def get_commentary_examples_with_fallback(commentary_bank, event_type, k=3, seed=42):
    """
    Retrieve up to k commentary examples for a given event type, with fallback.

    If no examples exist for the requested event type, fall back to the 'other' event type.

    Args:
        commentary_bank (dict): Dictionary mapping event types (str) to lists of commentary strings.
        event_type (str): The event type to retrieve examples for.
        k (int, optional): Maximum number of examples to return. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list of str: Up to k commentary examples for the event type, or for 'other' if none exist.

    Example:
        >>> get_commentary_examples_with_fallback({'other': ['Generic comment.']}, 'rare_event')
        ['Generic comment.']
    """
    examples = get_commentary_examples(commentary_bank, event_type, k=k, seed=seed)

    if examples:
        return examples

    # Fallback to 'other' event type if none found
    return get_commentary_examples(commentary_bank, "other", k=k, seed=seed)