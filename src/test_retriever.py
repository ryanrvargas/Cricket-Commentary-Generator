"""
test_retriever.py
-----------------
This script tests the retrieval of example commentary lines for different cricket event types.
It demonstrates how to use the commentary bank and retriever logic to fetch sample commentary
for a set of fixed event labels.

How it works:
    - Loads a commentary bank from the training CSV using build_commentary_bank.
    - For each test event label, retrieves up to 3 example commentary lines using get_commentary_examples.
    - Prints the event label and the retrieved examples for inspection.

Intended usage:
    Run this script to verify that the commentary retrieval logic is working and that the commentary
    bank contains reasonable examples for each event type. This is useful for debugging and for
    understanding the diversity of commentary available for each event.

Example output:
    === boundary_four ===
    1. Four runs. The batter finds the boundary.
    2. ...
    ...
"""

from load_commentary import build_commentary_bank
from retriever import get_commentary_examples

bank = build_commentary_bank("raw/train.csv")

test_labels = [
    "boundary_four",
    "wicket_caught",
    "single",
    "wide",
    "bye_or_legbye"
]

for label in test_labels:
    print(f"\n=== {label} ===")
    examples = get_commentary_examples(bank, label, k=3)

    for i, example in enumerate(examples, start=1):
        print(f"{i}. {example}")