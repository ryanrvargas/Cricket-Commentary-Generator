"""
finetune_t5.py
---------------
Fine-tune a small pretrained text-to-text model on cricket commentary pairs.

This version supports event-balanced training. Balanced training helps prevent
common event types from drowning out rarer but important event types such as
wickets.

Recommended install:
    pip install pandas torch transformers sentencepiece accelerate

Example balanced laptop run:
    python src/finetune_t5.py --output-dir models/t5-cricket-commentary-balanced --balance-train --samples-per-event 150 --epochs 3 --shuffle
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Let this script work from either the project root or inside src/.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.training.build_supervised_pairs import build_pairs_from_csv

def _require_transformers():
    """
    Import Hugging Face pieces with a clear message if dependencies are missing.
    """
    try:
        from transformers import (  # type: ignore
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing fine-tuning dependencies. Install them with:\n"
            "pip install pandas torch transformers sentencepiece accelerate"
        ) from exc

    return (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )


class CommentaryPairsDataset(Dataset):
    """
    Tokenized prompt-target pairs for seq2seq training.
    """

    def __init__(
        self,
        pairs: list[dict[str, Any]],
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.examples: list[dict[str, Any]] = []

        for pair in pairs:
            model_inputs = tokenizer(
                pair["input_text"],
                max_length=max_input_length,
                truncation=True,
            )

            labels = tokenizer(
                text_target=pair["target_text"],
                max_length=max_target_length,
                truncation=True,
            )

            model_inputs["labels"] = labels["input_ids"]
            self.examples.append(model_inputs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def _event_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(pair.get("event_type", "other") for pair in pairs).items()))


def _print_event_counts(title: str, pairs: list[dict[str, Any]]) -> None:
    print(f"\n{title} event counts:")
    for event_type, count in _event_counts(pairs).items():
        print(f"  {event_type}: {count}")


def _balance_pairs(
    pairs: list[dict[str, Any]],
    *,
    total_limit: int,
    samples_per_event: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Build a balanced training set by sampling each event type.

    If an event bucket has fewer examples than requested, sample with replacement.
    This is intentional for rare event types such as wickets.
    """
    if not pairs:
        return pairs

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[pair.get("event_type", "other")].append(pair)

    event_types = sorted(groups)
    if samples_per_event <= 0:
        if total_limit and total_limit > 0:
            samples_per_event = max(1, total_limit // len(event_types))
        else:
            samples_per_event = min(len(group) for group in groups.values())

    balanced: list[dict[str, Any]] = []
    for event_type in event_types:
        group = groups[event_type]
        if len(group) >= samples_per_event:
            balanced.extend(rng.sample(group, samples_per_event))
        else:
            balanced.extend(rng.choices(group, k=samples_per_event))

    rng.shuffle(balanced)
    if total_limit and total_limit > 0:
        balanced = balanced[:total_limit]

    return balanced


def _maybe_limit(
    pairs: list[dict[str, Any]],
    limit: int,
    *,
    seed: int,
    shuffle: bool,
) -> list[dict[str, Any]]:
    """
    Optionally shuffle and limit a split for faster experiments.
    """
    if shuffle:
        pairs = list(pairs)
        random.Random(seed).shuffle(pairs)

    if limit and limit > 0:
        return pairs[:limit]

    return pairs


def _load_split(path: str | Path, limit: int, *, seed: int, shuffle: bool) -> list[dict[str, Any]]:
    pairs = build_pairs_from_csv(path)
    return _maybe_limit(pairs, limit, seed=seed, shuffle=shuffle)


def _make_training_args(Seq2SeqTrainingArguments: Any, args: argparse.Namespace) -> Any:
    """
    Build training args while supporting both older and newer Transformers names.
    """
    common_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=args.fp16,
        report_to="none",
    )

    try:
        return Seq2SeqTrainingArguments(**common_kwargs, eval_strategy="epoch")
    except TypeError:
        return Seq2SeqTrainingArguments(**common_kwargs, evaluation_strategy="epoch")


def _save_metrics(metrics: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        clean_metrics[key] = value

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5/FLAN-T5 on cricket commentary prompt-target pairs."
    )
    parser.add_argument("--train-csv", default="raw/train.csv")
    parser.add_argument("--validation-csv", default="raw/validation.csv")
    parser.add_argument("--test-csv", default="raw/test.csv")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--output-dir", default="models/t5-cricket-commentary-best")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 on compatible GPUs.")
    parser.add_argument("--balance-train", action="store_true", help="Sample a more balanced training set across event types.")
    parser.add_argument("--samples-per-event", type=int, default=0, help="Examples per event type when --balance-train is used.")
    parser.add_argument("--show-event-counts", action="store_true", help="Print event type counts before training.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    ) = _require_transformers()

    set_seed(args.seed)

    print("Loading supervised pairs...")

    # For balanced training, load the full training set first, balance it, then
    # optionally cap it. For eval/test, keep the natural distribution.
    if args.balance_train:
        train_pairs_raw = build_pairs_from_csv(args.train_csv)
        if args.show_event_counts:
            _print_event_counts("Raw train", train_pairs_raw)
        train_pairs = _balance_pairs(
            train_pairs_raw,
            total_limit=args.max_train_samples,
            samples_per_event=args.samples_per_event,
            seed=args.seed,
        )
        if args.shuffle:
            random.Random(args.seed).shuffle(train_pairs)
    else:
        train_pairs = _load_split(
            args.train_csv,
            args.max_train_samples,
            seed=args.seed,
            shuffle=args.shuffle,
        )

    eval_pairs = _load_split(
        args.validation_csv,
        args.max_eval_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    test_pairs = _load_split(
        args.test_csv,
        args.max_test_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    if not train_pairs:
        raise SystemExit("No training pairs found. Check --train-csv.")
    if not eval_pairs:
        raise SystemExit("No validation pairs found. Check --validation-csv.")

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(eval_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    if args.show_event_counts:
        _print_event_counts("Final train", train_pairs)
        _print_event_counts("Validation", eval_pairs)
        _print_event_counts("Test", test_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_dataset = CommentaryPairsDataset(
        train_pairs,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )
    eval_dataset = CommentaryPairsDataset(
        eval_pairs,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )
    test_dataset = (
        CommentaryPairsDataset(
            test_pairs,
            tokenizer,
            args.max_input_length,
            args.max_target_length,
        )
        if test_pairs
        else None
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = _make_training_args(Seq2SeqTrainingArguments, args)

    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    print("Starting fine-tuning...")
    train_result = trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics: dict[str, Any] = {}
    metrics.update({f"train_{key}": value for key, value in train_result.metrics.items()})

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="validation")
    metrics.update(eval_metrics)

    if test_dataset is not None and len(test_dataset) > 0:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        metrics.update(test_metrics)

    metrics["model_name"] = args.model_name
    metrics["train_pairs"] = len(train_pairs)
    metrics["validation_pairs"] = len(eval_pairs)
    metrics["test_pairs"] = len(test_pairs)
    metrics["balance_train"] = bool(args.balance_train)
    metrics["samples_per_event"] = int(args.samples_per_event)
    metrics["train_event_counts"] = _event_counts(train_pairs)

    _save_metrics(metrics, args.output_dir)

    print(f"Done. Model saved to {args.output_dir}")
    print(f"Metrics saved to {Path(args.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
