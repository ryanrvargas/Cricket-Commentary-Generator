# Real-Time Sports Commentary Generator

A sports commentary generation project built around structured match events, commentary retrieval, and controlled text generation.

This repository currently contains:
- a **cricket-first TF-IDF + template baseline** that is the strongest and most presentation-ready system
- a **soccer expansion branch** using StatsBomb-style event data and a small commentary bank
- an **experimental T5 fine-tuning branch** for neural commentary generation

The assignment goal is broader than the current strongest baseline. The repo now supports both:
1. a reliable retrieval-based system for live-style commentary demos
2. a neural fine-tuning path for experimentation and comparison

---

## Current Project Layout

```text
Cricket-Commentary-Generator/
├── models/
├── models-backup/
├── raw/
├── raw_soccer/
│   ├── statsbomb/
│   │   ├── events/
│   │   ├── lineups/
│   │   └── matches/
│   └── yallashoot/
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── providers.py
│   │   ├── retriever.py
│   │   └── tfidf_vectorizer.py
│   ├── cricket/
│   │   ├── __init__.py
│   │   ├── classify_event.py
│   │   ├── event_types.py
│   │   ├── generator.py
│   │   ├── load_commentary.py
│   │   ├── load_events.py
│   │   ├── stream_demo.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_generator.py
│   │       ├── test_generator_quality.py
│   │       ├── test_generator_restraint.py
│   │       ├── test_loader.py
│   │       └── test_retriever.py
│   ├── soccer/
│   │   ├── __init__.py
│   │   ├── classify_soccer_event.py
│   │   ├── load_soccer_commentary.py
│   │   ├── load_soccer_events.py
│   │   ├── soccer_generator.py
│   │   ├── stream_demo_soccer.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_soccer_loader.py
│   └── training/
│       ├── __init__.py
│       ├── build_supervised_pairs.py
│       ├── build_supervised_pairs_controlled.py
│       ├── build_supervised_pairs_multisport.py
│       ├── build_supervised_pairs_soccer.py
│       ├── finetune_t5.py
│       ├── model_infer.py
│       └── t5_stream_demo.py
```

---

## What Each Package Does

### `src/common/`
Shared infrastructure used by both sports.

- **`providers.py`**  
  Event-provider abstraction. Supports:
  - historical replay
  - timed pseudo-live replay
  - live-provider stub for future API integration

- **`retriever.py`**  
  Retrieves commentary examples by event type and ranks them with in-house TF-IDF.

- **`tfidf_vectorizer.py`**  
  Small custom TF-IDF implementation with cosine similarity.

### `src/cricket/`
Core cricket pipeline.

- **`event_types.py`**  
  Defines the shared fixed cricket label set.

- **`classify_event.py`**  
  Maps one flattened Cricsheet delivery into a fixed event label.

- **`load_events.py`**  
  Loads and flattens Cricsheet JSON into one event dictionary per delivery.

- **`load_commentary.py`**  
  Parses commentary CSV rows and builds a commentary bank grouped by event type.

- **`generator.py`**  
  Generates short event-faithful commentary using controlled template families and score context.

- **`stream_demo.py`**  
  Main polished cricket demo. This is the strongest version of the project.

- **`tests/`**  
  Support and quality checks for loading, retrieval, generator quality, and restraint.

### `src/soccer/`
Soccer expansion branch.

- **`classify_soccer_event.py`**  
  Maps flattened StatsBomb-style events into a fixed soccer label set.

- **`load_soccer_events.py`**  
  Loads and flattens StatsBomb event JSON.

- **`load_soccer_commentary.py`**  
  Loads soccer commentary CSV data and groups commentary lines into buckets.

- **`soccer_generator.py`**  
  Generates short soccer commentary with controlled templates.

- **`stream_demo_soccer.py`**  
  Runs the soccer demo over historical or timed replay data.

- **`tests/`**  
  Soccer loader test utilities.

### `src/training/`
Neural fine-tuning and inference utilities.

- **`build_supervised_pairs.py`**  
  Builds cricket prompt-target training pairs using a simplified prompt schema.

- **`build_supervised_pairs_controlled.py`**  
  Builds controlled canonical cricket targets for more stable T5 learning.

- **`build_supervised_pairs_soccer.py`**  
  Builds controlled soccer prompt-target pairs.

- **`build_supervised_pairs_multisport.py`**  
  Combines prebuilt pair CSVs into one multisport training file.

- **`finetune_t5.py`**  
  Fine-tunes a small pretrained seq2seq model such as FLAN-T5.

- **`model_infer.py`**  
  Runs one-off inference with optional event fact guardrails.

- **`t5_stream_demo.py`**  
  Runs a full cricket T5 demo over a saved match file.

---

## Data Files

### Cricket
Place cricket data in `raw/`.

Expected files:
- `raw/train.csv`
- `raw/validation.csv`
- `raw/test.csv`
- match files such as:
  - `raw/1527574.json`
  - `raw/1527575.json`
  - `raw/1527576.json`

### Soccer
Place soccer data in `raw_soccer/`.

Expected folders:
- `raw_soccer/statsbomb/events/`
- `raw_soccer/statsbomb/lineups/`
- `raw_soccer/statsbomb/matches/`
- `raw_soccer/yallashoot/`

Expected soccer commentary files:
- `raw_soccer/yallashoot/commentary_train.csv`
- `raw_soccer/yallashoot/commentary_train_augmented.csv`

---

## Setup

Install dependencies:

```bash
pip install pandas torch transformers sentencepiece accelerate pytest
```

If you only want the TF-IDF demos and not the neural model:

```bash
pip install pandas pytest
```

---

## Main Demo Commands

Run these from the **project root**.

### Cricket TF-IDF baseline
```bash
python -m src.cricket.stream_demo --max-events 24
```

### Cricket TF-IDF debug mode
```bash
python -m src.cricket.stream_demo --max-events 8 --debug
```

### Cricket TF-IDF pseudo-live replay
```bash
python -m src.cricket.stream_demo --max-events 24 --provider timed --delay 0.5
```

### Soccer TF-IDF demo
```bash
python -m src.soccer.stream_demo_soccer --commentary-csv raw_soccer/yallashoot/commentary_train_augmented.csv --max-events 15
```

### Soccer TF-IDF debug mode
```bash
python -m src.soccer.stream_demo_soccer --commentary-csv raw_soccer/yallashoot/commentary_train_augmented.csv --max-events 15 --debug
```

### Soccer TF-IDF demo including passes
```bash
python -m src.soccer.stream_demo_soccer --commentary-csv raw_soccer/yallashoot/commentary_train_augmented.csv --max-events 15 --include-passes
```

---

## T5 / Neural Demo Commands

### T5 cricket demo using the current best checkpoint
```bash
python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-best --max-events 24
```

### T5 cricket demo with event labels shown
```bash
python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-best --max-events 24 --show-event-type
```

### T5 cricket demo with prompts shown
```bash
python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-best --max-events 8 --show-prompts --show-event-type
```

### T5 controlled cricket demo
```bash
python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-controlled --max-events 24 --show-event-type
```

### T5 newer controlled checkpoint demo
```bash
python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-controlled-v5 --max-events 24 --show-event-type
```

### T5 multisport demo
```bash
python -m src.training.t5_stream_demo --checkpoint models/t5-multisport-controlled --max-events 24 --show-event-type
```

### One-off T5 inference for a specific event
```bash
python -m src.training.model_infer --checkpoint models/t5-cricket-commentary-best --match raw/1527575.json --event-index 13 --show-prompt
```

### One-off T5 inference without fact guard
```bash
python -m src.training.model_infer --checkpoint models/t5-cricket-commentary-best --match raw/1527575.json --event-index 13 --no-fact-guard --show-prompt
```

---

## Training Commands

### Build default cricket supervised pairs
```bash
python -m src.training.build_supervised_pairs --input raw/train.csv --output raw/train_pairs.csv --shuffle
```

### Build controlled cricket supervised pairs
```bash
python -m src.training.build_supervised_pairs_controlled --input raw/train.csv --output raw/train_pairs_controlled.csv --shuffle
```

### Build soccer supervised pairs
```bash
python -m src.training.build_supervised_pairs_soccer --input raw_soccer/yallashoot/commentary_train_augmented.csv --output raw_soccer/yallashoot/soccer_pairs_controlled.csv --shuffle --show-event-counts
```

### Combine cricket + soccer pair CSVs into one multisport file
```bash
python -m src.training.build_supervised_pairs_multisport --input raw/train_pairs_controlled.csv raw_soccer/yallashoot/soccer_pairs_controlled.csv --output raw/multisport_pairs_controlled.csv --shuffle
```

### Fine-tune T5 on controlled cricket pairs
```bash
python -m src.training.finetune_t5 --output-dir models/t5-cricket-commentary-controlled --balance-train --samples-per-event 100 --epochs 2 --shuffle --show-event-counts --max-eval-samples 1000 --max-test-samples 1000 --eval-batch-size 8 --train-batch-size 4 --learning-rate 3e-5
```

### Fine-tune T5 on a prebuilt multisport pair CSV
```bash
python -m src.training.finetune_t5 --train-pairs-csv raw/multisport_pairs_controlled.csv --validation-pairs-csv raw/validation_pairs_controlled.csv --test-pairs-csv raw/test_pairs_controlled.csv --output-dir models/t5-multisport-controlled-working --epochs 2 --shuffle --train-batch-size 4 --eval-batch-size 8 --learning-rate 3e-5
```

---

## Recommended Presentation Flow

Use this order for a live demo:

1. **Cricket TF-IDF baseline**
   ```bash
   python -m src.cricket.stream_demo --max-events 24
   ```

2. **Soccer expansion**
   ```bash
   python -m src.soccer.stream_demo_soccer --commentary-csv raw_soccer/yallashoot/commentary_train_augmented.csv --max-events 12 --debug
   ```

3. **T5 experimental branch**
   ```bash
   python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-controlled-v5 --max-events 12 --show-event-type
   ```

This gives the strongest honest story:
- polished cricket baseline
- multi-sport expansion
- pretrained/fine-tuned neural branch

---

## Current Strengths

- Structured event ingestion works for cricket and soccer.
- Cricket TF-IDF retrieval and generation are the strongest part of the repo.
- The provider abstraction supports historical replay, timed replay, and a live stub.
- The package layout is now organized by shared utilities, sport-specific logic, and training code.

---

## Current Limitations

- The strongest production-like system is still the **cricket TF-IDF baseline**, not the T5 model.
- Soccer commentary retrieval depends on a small and partly synthetic commentary bank.
- The live provider is currently a stub and does not yet poll a real sports API.
- The T5 branch is experimental and should be presented as an extension, not the most reliable system.

---

## Suggested Final Verification

```bash
python -m py_compile src/common/providers.py src/common/retriever.py src/common/tfidf_vectorizer.py src/cricket/load_events.py src/cricket/classify_event.py src/cricket/load_commentary.py src/cricket/generator.py src/cricket/stream_demo.py src/soccer/load_soccer_events.py src/soccer/classify_soccer_event.py src/soccer/load_soccer_commentary.py src/soccer/soccer_generator.py src/soccer/stream_demo_soccer.py src/training/build_supervised_pairs.py src/training/build_supervised_pairs_controlled.py src/training/build_supervised_pairs_soccer.py src/training/build_supervised_pairs_multisport.py src/training/finetune_t5.py src/training/model_infer.py src/training/t5_stream_demo.py

pytest src/cricket/tests

python -m src.cricket.stream_demo --max-events 24
python -m src.soccer.stream_demo_soccer --commentary-csv raw_soccer/yallashoot/commentary_train_augmented.csv --max-events 12 --debug
python -m src.training.t5_stream_demo --checkpoint models/t5-cricket-commentary-controlled-v5 --max-events 12 --show-event-type
```

---

## Project Scope Statement

This repository now contains:
- a completed cricket retrieval-based commentary pipeline
- a working soccer expansion branch
- a T5 fine-tuning path for neural commentary experiments
- a provider layer that supports historical replay and timed pseudo-live demos

The broad class prompt asks for pretrained fine-tuning, diverse sports, and live-style data ingestion. This repo moves toward all three, but the most reliable finished demo remains the cricket TF-IDF + controlled-generation baseline.
