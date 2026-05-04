"""
finetune_t5.py
---------------
Fine-tune a small pretrained text-to-text model on cricket commentary pairs.

This script loads train.csv, validation.csv, and test.csv, converts each file
into prompt-target pairs using build_supervised_pairs.py, fine-tunes a T5-style
model, and saves the trained checkpoint plus evaluation losses.

Recommended install:
    pip install pandas torch transformers sentencepiece accelerate

Example quick smoke run on a laptop:
    python src/finetune_t5.py --max-train-samples 500 --max-eval-samples 100 --epochs 1

Example fuller run:
    python src/finetune_t5.py --model-name google/flan-t5-small --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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

from build_supervised_pairs import build_pairs_from_csv  # noqa: E402


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
        return Seq2SeqTrainingArguments(
            **common_kwargs,
            eval_strategy="epoch",
        )
    except TypeError:
        return Seq2SeqTrainingArguments(
            **common_kwargs,
            evaluation_strategy="epoch",
        )


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
    parser.add_argument("--output-dir", default="models/t5-cricket-commentary")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 on compatible GPUs.")
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
    test_dataset = CommentaryPairsDataset(
        test_pairs,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    ) if test_pairs else None

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = _make_training_args(Seq2SeqTrainingArguments, args)

    # Newer Transformers versions renamed the Trainer argument from
    # tokenizer= to processing_class=. Keep a fallback so the script works
    # across both newer and older installs.
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

    _save_metrics(metrics, args.output_dir)

    print(f"Done. Model saved to {args.output_dir}")
    print(f"Metrics saved to {Path(args.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
