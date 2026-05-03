# Cricket Real-Time Commentary Generator

This project is a cricket-focused prototype for a real-time sports commentary generator. It reads ball-by-ball cricket match data, classifies each delivery into a cricket event type, retrieves similar commentary examples from a commentary corpus, and generates one short commentary line for each ball.

The project simulates real-time commentary by replaying saved Cricsheet JSON match files one delivery at a time. It does not use a live sports API yet, and it does not fine-tune a neural language model. Instead, it uses a controlled pipeline made of event classification, TF-IDF retrieval, and template-based generation.

## What the Project Does

The system takes structured cricket data like this:

- batter
- bowler
- runs scored
- extras
- wickets
- innings
- over and ball number

Then it produces commentary like this:

```text
0.3      Mohammad Rizwan vs Mohammad Ali
         Mohammad Rizwan leans into the drive and finds the boundary. That moves them to 4/0.
```

The goal is to turn match events into readable cricket commentary while keeping the generated text factually tied to the match data.

## Project Pipeline

The project works in this order:

1. Load a Cricsheet JSON match file.
2. Flatten the nested innings, overs, and deliveries into simple event dictionaries.
3. Classify each delivery into a fixed event type.
4. Load cricket commentary examples from `train.csv`.
5. Build a commentary bank grouped by event type.
6. Retrieve relevant examples using event filtering and in-house TF-IDF ranking.
7. Generate one commentary line for the delivery.
8. Stream the match through `stream_demo.py` as a final demo.

## Event Types

The project uses these fixed event labels:

```text
dot_ball
single
double
triple
boundary_four
boundary_six
wicket_bowled
wicket_caught
wicket_lbw
run_out
wide
no_ball
bye_or_legbye
other
```

These labels are shared across the loader, classifier, retriever, generator, and tests.

## Repository Structure

```text
Cricket-Commentary-Generator/
│   documentation.md
│   README.md
│   Repo-Tree.txt
│
├── raw/
│   │   1527574.json
│   │   1527575.json
│   │   1527576.json
│   │   1529145.json
│   │   1529266.json
│   │   1529267.json
│   │   README.txt
│   │   train.csv
│   │   validation.csv
│   │   test.csv
│
└── src/
    │   classify_event.py
    │   event_types.py
    │   generator.py
    │   load_commentary.py
    │   load_events.py
    │   retriever.py
    │   stream_demo.py
    │   tfidf_vectorizer.py
    │
    │   test_generator.py
    │   test_generator_quality.py
    │   test_generator_restraint.py
    │   test_loader.py
    │   test_retriever.py
```

## Main Files

| File | Purpose |
|---|---|
| `src/load_events.py` | Loads and flattens Cricsheet JSON match data. |
| `src/classify_event.py` | Classifies each ball into a fixed cricket event label. |
| `src/event_types.py` | Stores the shared event label list. |
| `src/load_commentary.py` | Parses the commentary CSV and builds the commentary bank. |
| `src/tfidf_vectorizer.py` | Implements a small in-house TF-IDF vectorizer. |
| `src/retriever.py` | Retrieves event-specific commentary examples. |
| `src/generator.py` | Generates final commentary lines using templates and retrieved style categories. |
| `src/stream_demo.py` | Runs the final clean match demo. |

## Requirements

This project uses Python and a small number of packages.

Recommended version:

```text
Python 3.12+
```

Install the needed packages:

```bash
pip install pandas pytest
```

No scikit-learn dependency is required because the project includes its own TF-IDF vectorizer.

## How to Run the Final Demo

From the project root, run:

```bash
python src/stream_demo.py
```

By default, the demo uses:

```text
raw/1527575.json
raw/train.csv
```

It shows the first 24 deliveries in a clean presentation format.

You can also run:

```bash
python src/stream_demo.py --max-events 24
```

To make the output feel more like a live feed, add a delay:

```bash
python src/stream_demo.py --max-events 24 --delay 0.5
```

To show the full match, use:

```bash
python src/stream_demo.py --max-events 0
```

## Running with a Different Match File

You can pass a different Cricsheet match JSON file:

```bash
python src/stream_demo.py --match raw/1527574.json --commentary-csv raw/train.csv --max-events 24
```

## Debug Mode

Normal mode is meant for the final presentation. It hides internal trace details.

Debug mode is available for testing:

```bash
python src/stream_demo.py --max-events 2 --debug
```

Debug mode prints details like:

- event label
- score after the ball
- runs on the ball
- generated commentary

Use debug mode when checking whether classification, scoring, and generation are working correctly.

## How to Run Tests

From the project root, run:

```bash
python -m pytest src
```

You can also run the specific quality tests:

```bash
python -m pytest src/test_generator_quality.py src/test_generator_restraint.py
```

These tests check that the generator avoids awkward phrasing, mixed cricket shot language, and repeated wording such as "keeps the strike moving to rotate the strike."

## Quick Verification Commands

Use these before submitting or presenting:

```bash
python -m py_compile src/load_events.py src/classify_event.py src/load_commentary.py src/retriever.py src/tfidf_vectorizer.py src/generator.py src/stream_demo.py
python -m pytest src
python src/stream_demo.py --max-events 24
python src/stream_demo.py --max-events 2 --debug
```

## Example Output

```text
Real-Time Cricket Commentary Demo
Match data: 1527575.json
Commentary corpus: train.csv
Showing first 24 deliveries
Mode: final demo
========================================================================

Innings 1 - Rawalpindiz
------------------------------------------------------------------------
0.1      Mohammad Rizwan vs Mohammad Ali
         Mohammad Rizwan meets Mohammad Ali with a firm defence. Still 0/0.
0.2      Mohammad Rizwan vs Mohammad Ali
         Mohammad Rizwan gets behind it and defends. No run. Still 0/0.
0.3      Mohammad Rizwan vs Mohammad Ali
         Mohammad Rizwan leans into the drive and finds the boundary. That moves them to 4/0.
```

## Current Status

The main project pipeline is complete:

- Cricsheet JSON loading works.
- Ball-by-ball event flattening works.
- Event classification works.
- Commentary CSV parsing works.
- Commentary retrieval works.
- Template-based generation works.
- The final stream demo is presentation-ready.
- Debug mode is still available for testing.

The remaining improvements are optional polish, not core blockers.

## Limitations

This project has a few intentional limits:

- It is cricket-only.
- It uses saved match files instead of a live sports API.
- It uses templates and retrieval instead of fine-tuning a large language model.
- It generates short ball-by-ball lines instead of long broadcast paragraphs.
- It does not yet track advanced cricket context like batter score, bowler figures, partnership, current run rate, or required run rate.

## Future Improvements

Possible next steps:

- Add batter score and bowler figures.
- Add current run rate and required run rate.
- Cache TF-IDF vectors instead of rebuilding them during retrieval.
- Use `validation.csv` and `test.csv` for more formal evaluation.
- Add a live cricket data feed.
- Add a small LLM rewriting layer after factual constraints are enforced.
- Expand the design to other sports after the cricket prototype is stable.

## Summary

This project demonstrates a working cricket commentary generator. It uses structured match data to keep the facts correct and uses a commentary corpus to guide the style of the generated text. The final demo shows how saved ball-by-ball match data can be replayed like a live commentary feed.
