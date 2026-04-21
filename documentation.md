# Cricket Commentary Generator: Project Overview

## Project Purpose
This project generates realistic cricket commentary for each ball of a match, using real match data and a bank of authentic commentary phrases. It is designed to:
- Parse raw Cricsheet match data (ball-by-ball JSON)
- Classify each delivery into a standardized event label (e.g., dot ball, single, four, wicket)
- Retrieve and sample real commentary lines for each event type
- Generate a plausible commentary sentence for every ball in a match


## High-Level Pipeline

1. **Raw Data Ingestion**
   - Reads Cricsheet JSON files (one per match) from the `raw/` directory.
2. **Event Flattening**
   - Flattens each delivery into a simple event dictionary (see `load_events.py`).
3. **Event Classification**
   - Assigns a standardized event label to each delivery (see `classify_event.py` and `event_types.py`).
4. **Commentary Retrieval**
   - Looks up and samples real commentary lines for the event label (see `retriever.py`).
5. **Commentary Generation**
   - Assembles/generates a commentary sentence for each ball (see `generator.py`).


## File-by-File Breakdown

### `src/event_types.py`
- **Purpose:** Defines the set of standardized event labels (e.g., dot_ball, single, four, wicket, etc.) used throughout the project.
- **Role:** Ensures all modules use a consistent vocabulary for event types.

### `src/load_events.py`
- **Purpose:** Reads a Cricsheet match JSON and flattens it into a list of simple event dictionaries (one per delivery).
- **Key Features:**
  - Handles complex nested JSON structure.
  - Tracks both raw delivery order (`delivery_index`) and legal ball number (`ball_in_over`), correctly handling extras (wides, no-balls).
- **Role:** Converts raw match data into a format suitable for classification and commentary generation.

### `src/classify_event.py`
- **Purpose:** Assigns a standardized event label to each flattened event row, using the definitions in `event_types.py`.
- **Role:** Bridges the gap between raw delivery data and commentary retrieval by labeling each ball.

### `src/generator.py`
- **Purpose:** Generates a commentary sentence for each event, using the event label and sampled commentary lines.
- **Role:** The main commentary generation logic; may combine templates, real examples, and event data.

### `src/retriever.py`
- **Purpose:** Retrieves and samples real commentary lines for a given event type from a commentary bank.
- **Key Functions:**
  - `get_commentary_examples`: Returns up to `k` random examples for a given event type.
  - `get_commentary_examples_with_fallback`: Falls back to generic examples if none exist for the event type.
- **Role:** Supplies realistic commentary text for each event label.

### `src/load_commentary.py`
- **Purpose:** Loads and processes the commentary corpus (e.g., from Kaggle or other sources) into a bank of phrases grouped by event label.
- **Role:** Prepares the commentary bank used by `retriever.py`.

### `src/test_*.py`
- **Purpose:** Unit tests for the main modules (e.g., `test_generator.py`, `test_loader.py`, `test_retriever.py`).
- **Role:** Ensures correctness and reliability of each pipeline stage.


## Data Folders

### `raw/`
- **Contains:**
  - Cricsheet match JSON files (e.g., `1527574.json`)
  - CSVs for training/validation (if using ML components)
  - README.txt for data notes
- **Role:** Source of all raw match and commentary data.


## How the Pieces Fit Together

1. **Start with a match JSON in `raw/`**
2. **`load_events.py`** flattens the match into a list of event dicts (one per ball)
3. **`classify_event.py`** assigns an event label to each event dict
4. **`load_commentary.py`** loads the commentary corpus and builds a commentary bank (grouped by event label)
5. **`retriever.py`** samples real commentary lines for each event label
6. **`generator.py`** combines the event data and commentary line to produce a final commentary sentence
7. **Tests** in `src/test_*.py` ensure each step works as expected


## Example Flow

1. **Input:** `raw/1527574.json` (Cricsheet match)
2. **Flatten:** `load_events.py` → list of deliveries (dicts)
3. **Classify:** `classify_event.py` → add event label to each delivery
4. **Commentary Bank:** `load_commentary.py` → build event label → commentary lines mapping
5. **Retrieve:** `retriever.py` → get sample commentary for each event label
6. **Generate:** `generator.py` → produce commentary sentence for each ball


## Summary Table

| File                  | Main Role                                      |
--|
| event_types.py        | Defines event label vocabulary                  |
| load_events.py        | Flattens match JSON to event dicts              |
| classify_event.py     | Assigns event label to each event               |
| load_commentary.py    | Loads and groups commentary corpus              |
| retriever.py          | Samples commentary lines for event labels       |
| generator.py          | Generates commentary sentences                  |
| test_*.py             | Unit tests for each module                      |


## Getting Started

1. Place Cricsheet match JSONs in `raw/`
2. Prepare a commentary corpus (e.g., Kaggle cricket commentary) and process with `load_commentary.py`
3. Use the pipeline (see above) to generate commentary for each ball in a match
4. Run tests in `src/` to verify correctness


