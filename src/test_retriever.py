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