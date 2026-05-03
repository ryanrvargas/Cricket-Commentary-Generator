# Cricket Real-Time Commentary Generator Documentation

## 1. Project Overview

This project is a cricket-focused prototype for a real-time sports commentary generator. It takes structured ball-by-ball cricket match data, classifies each delivery into a fixed event type, retrieves similar cricket commentary examples, and generates a short commentary line for each ball.

The larger assignment describes a real-time sports commentary system that analyzes live game data and uses sports commentary text to generate natural commentary. This implementation narrows that broad idea into a working cricket pipeline. Instead of trying to support every sport or fine-tune a large model immediately, this project focuses on a complete end-to-end cricket version that can be demonstrated and tested.

The current system can:

- Read Cricsheet cricket match JSON files.
- Flatten nested ball-by-ball match data into simple event dictionaries.
- Classify each ball as a dot ball, single, boundary, wicket, extra, or other cricket event.
- Parse a cricket commentary CSV dataset.
- Build a commentary bank grouped by event type.
- Retrieve event-specific commentary examples using simple filtering and TF-IDF ranking.
- Generate one commentary line per ball using templates, retrieved style hints, and live score context.
- Replay a saved match sequentially to simulate real-time commentary.

## 2. Assignment Alignment

The assignment asks for three main pieces:

| Assignment Requirement | How This Project Handles It |
|---|---|
| Analyze sports data and statistics | Uses Cricsheet ball-by-ball JSON data with batters, bowlers, runs, extras, wickets, innings, and overs. |
| Use commentary text as the language source | Uses the Kaggle-style cricket commentary CSV files to build a bank of real commentary phrases. |
| Generate real-time commentary | Replays saved match data one delivery at a time through `stream_demo.py`, producing one commentary line per ball. |

This project does **not** currently fine-tune a neural language model. That is a future improvement. The current version uses a lighter and more reliable design: structured event classification, commentary retrieval, and template-based generation. This makes the system easier to test and demo while still matching the core idea of turning live-style sports events into commentary.

## 3. Project Scope

The original assignment is broad because it mentions diverse sports, large datasets, pretraining, fine-tuning, and live game data feeds. For this version, the project scope is intentionally smaller:

- Sport: cricket only.
- Data source: saved Cricsheet JSON match files.
- Commentary source: cricket commentary CSV files.
- Real-time behavior: simulated by streaming saved ball-by-ball data in order.
- Generation method: template-based generation with retrieved commentary style hints.

This is a practical scope decision. A multi-sport fine-tuned model would be much harder to finish and validate. The current cricket pipeline gives a complete working system first, then leaves room for future model improvements.

## 4. Repository Structure

Expected project layout:

```text
Cricket-Commentary-Generator/
│   .gitattributes
│   .gitignore
│   documentation.md
│   Repo-Tree.txt
│
├───raw/
│       1527574.json
│       1527575.json
│       1527576.json
│       1529145.json
│       1529266.json
│       1529267.json
│       README.txt
│       test.csv
│       train.csv
│       validation.csv
│
└───src/
        classify_event.py
        event_types.py
        generator.py
        load_commentary.py
        load_events.py
        retriever.py
        stream_demo.py
        test_generator.py
        test_loader.py
        test_retriever.py
        tfidf_vectorizer.py
```

## 5. Data Sources

### 5.1 Cricsheet Match JSON Files

The `raw/` folder contains Cricsheet JSON files. These files store cricket matches in a nested format with innings, overs, and deliveries. Each delivery can include fields such as:

- batter
- bowler
- non-striker
- runs scored by the batter
- extras
- wides
- no-balls
- byes
- leg byes
- wicket information

The included Cricsheet README describes six recently added male cricket matches, including PSL, IPL, and international T20 matches. These match files act as the structured event data for the commentary generator.

### 5.2 Commentary CSV Files

The project also includes:

- `train.csv`
- `validation.csv`
- `test.csv`

These CSV files contain cricket commentary text and related metadata. The current implementation mainly uses `train.csv` to build the commentary bank. The commentary CSV stores information in a `rows` column, so `load_commentary.py` parses that text into structured fields before grouping commentary lines by event type.

## 6. Core Design Idea

The project separates **match truth** from **commentary style**.

### Match Truth

Match truth comes from Cricsheet JSON files. These files determine what actually happened on each ball:

- Was it a dot ball?
- Was it a single?
- Was it a four?
- Was there a wicket?
- Was the delivery a wide or no-ball?

This is handled by `load_events.py` and `classify_event.py`.

### Commentary Style

Commentary style comes from the commentary CSV dataset. The commentary text is used to provide realistic phrasing, but it does not decide what happened in the match. It only supports retrieval and style hints.

This is handled by `load_commentary.py`, `retriever.py`, `tfidf_vectorizer.py`, and `generator.py`.

This separation is important because the system should not invent the event. The structured match data controls the facts, and the commentary text helps the wording sound more natural.

## 7. Event Labels

The project uses one fixed set of event labels across the entire pipeline. These labels are defined in `src/event_types.py`:

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

Every major module depends on this shared vocabulary. The event loader creates event rows, the classifier assigns one of these labels, the commentary bank groups phrases by these labels, the retriever searches inside these labels, and the generator uses them to choose templates.

## 8. High-Level Pipeline

### Step 1: Load Match Data

`load_events.py` reads a Cricsheet JSON match file from `raw/`.

Example:

```python
events = load_match_events("raw/1527575.json")
```

The output is a list of event dictionaries, one dictionary per delivery.

### Step 2: Flatten Deliveries

Cricsheet JSON is nested by innings, overs, and deliveries. The project flattens each delivery into a simpler dictionary with fields like:

```text
match_id
innings
batting_team
over
delivery_index
ball_in_over
batter
bowler
non_striker
runs_off_bat
extras
wides
noballs
byes
legbyes
wicket_type
player_dismissed
```

This makes later classification and generation much easier.

### Step 3: Track Legal Ball Numbers

`load_events.py` tracks both:

- `delivery_index`: raw delivery order inside the over.
- `ball_in_over`: legal ball number inside the over.

This distinction matters because wides and no-balls are real deliveries, but they do not count as legal balls in cricket. For example, if a wide happens on ball 4.5, the next legal ball should still become 4.6 rather than incorrectly moving to 4.7.

### Step 4: Classify Each Event

`classify_event.py` maps each flattened delivery into one event label.

The classification order is:

1. Wickets
2. Extras
3. Scoring shots
4. Dot balls
5. Other fallback

Examples:

```text
wicket_type == "bowled"      -> wicket_bowled
wicket_type == "caught"      -> wicket_caught
wides > 0                    -> wide
noballs > 0                  -> no_ball
byes > 0 or legbyes > 0      -> bye_or_legbye
runs_off_bat == 6            -> boundary_six
runs_off_bat == 4            -> boundary_four
runs_off_bat == 1            -> single
runs_off_bat == 0, extras 0  -> dot_ball
```

### Step 5: Load and Parse Commentary Text

`load_commentary.py` reads the commentary CSV. The CSV is expected to have a `rows` column.

The module:

1. Parses each raw row string.
2. Extracts fields such as play type, teams, runs, bowler, batter, dismissal type, and commentary.
3. Cleans the commentary text by removing HTML and normalizing whitespace.
4. Maps each commentary row into one of the fixed event labels.
5. Builds a commentary bank grouped by event type.

The final commentary bank looks like this conceptually:

```python
{
    "boundary_four": ["...", "..."],
    "wicket_caught": ["...", "..."],
    "single": ["...", "..."],
    "dot_ball": ["...", "..."],
}
```

### Step 6: Retrieve Commentary Examples

`retriever.py` retrieves up to `k` commentary examples for the current event type.

The retriever works in two stages:

1. Filter commentary examples by event label.
2. If event data is provided, rank examples using the in-house TF-IDF vectorizer.

The query includes event information such as:

- event type
- batter
- bowler
- runs off the bat
- wicket type
- over number

This means the retrieval system is not just random sampling. It tries to rank examples that are more relevant to the current event.

### Step 7: Generate Commentary

`generator.py` creates the final commentary line.

It uses:

- the classified event type
- the batter and bowler names
- the dismissed player, when there is a wicket
- retrieved commentary examples
- small style hints from the retrieved examples
- optional live score context

For example, a `boundary_four` event might produce a line such as:

```text
Beautifully timed by Mohammad Rizwan, and that is four with a crisp drive.
```

A `wicket_caught` event might produce something like:

```text
Caught. Yasir Khan has to go as the catch is taken safely.
```

### Step 8: Stream the Match Demo

`stream_demo.py` ties the pieces together. It loads a match, builds the commentary bank, processes each event in order, updates the innings score, and prints generated commentary.

The demo simulates real-time commentary by replaying a saved match sequentially. It also tracks:

- current innings
- innings score
- wickets lost
- runs on the ball
- whether the ball ended the over

## 9. File-by-File Breakdown

| File | Purpose |
|---|---|
| `src/event_types.py` | Defines the fixed event label vocabulary used across the project. |
| `src/load_events.py` | Loads and flattens Cricsheet JSON match files into event dictionaries. |
| `src/classify_event.py` | Classifies each flattened delivery into a cricket event label. |
| `src/load_commentary.py` | Parses commentary CSV rows and builds a commentary bank grouped by event type. |
| `src/tfidf_vectorizer.py` | Implements tokenization, TF-IDF vectorization, and cosine similarity without scikit-learn. |
| `src/retriever.py` | Retrieves and ranks commentary examples for a given event type. |
| `src/generator.py` | Generates final commentary lines using templates, event facts, retrieved examples, and score context. |
| `src/stream_demo.py` | Runs the full sequential match demo and prints commentary line by line. |
| `src/test_loader.py` | Prints loaded and classified events to verify flattening and edge cases. |
| `src/test_retriever.py` | Checks that real match events can retrieve examples for selected labels. |
| `src/test_generator.py` | Generates commentary for early match events to verify the generation path. |

## 10. Module Details

### 10.1 `event_types.py`

This file stores the official event labels in one list called `EVENT_TYPES`. Keeping labels centralized prevents spelling mismatches across files.

### 10.2 `load_events.py`

This file is responsible for turning Cricsheet JSON into usable event rows.

Important behavior:

- Reads JSON with `json.load`.
- Uses the filename stem as `match_id`.
- Loops through innings, overs, and deliveries.
- Pulls out runs, extras, and wicket information.
- Stores only the first wicket on a delivery if multiple wickets exist.
- Increments `ball_in_over` only for legal deliveries.

This module is the foundation of the real-time analysis component because it turns raw match data into structured events the rest of the system can process.

### 10.3 `classify_event.py`

This file classifies a single delivery dictionary.

Its most important design choice is the classification order. Wickets are checked before normal scoring shots. That matters because a ball can include both runs and a wicket. Extras are also checked before standard batting runs so wides, no-balls, byes, and leg byes get labeled correctly.

### 10.4 `load_commentary.py`

This file prepares the language side of the project.

It does not use commentary text to decide what happened in the match. Instead, it uses commentary text to build event-specific phrase groups. This keeps factual event detection separate from language generation.

Key functions:

- `parse_commentary_row(raw_row)`
- `map_commentary_to_event_type(parsed_row)`
- `load_commentary_rows(csv_path)`
- `build_commentary_bank(csv_path)`

### 10.5 `tfidf_vectorizer.py`

This file implements a small TF-IDF system directly in the project.

It includes:

- `tokenize(text)`
- `TfidfVectorizerInHouse.fit()`
- `TfidfVectorizerInHouse.transform_one()`
- `TfidfVectorizerInHouse.fit_transform()`
- `cosine_similarity_sparse(vec_a, vec_b)`

This avoids requiring scikit-learn and keeps the project lightweight.

### 10.6 `retriever.py`

This file retrieves commentary examples from the commentary bank.

If only an event label is provided, it returns the first `k` examples from that event bucket. If full event data is provided, it builds a small query and ranks commentary examples using TF-IDF cosine similarity.

This helps the generator use commentary examples that are closer to the current ball.

### 10.7 `generator.py`

This file creates the final commentary sentence.

The generator uses templates for each event type. It then optionally adds a style hint based on the top retrieved commentary example.

For example:

- If a retrieved `boundary_four` example mentions a drive, the generator may add “with a crisp drive.”
- If a retrieved wicket example mentions an edge, the generator may add “after finding the edge.”
- If context is provided, the generator adds score context such as “They move to 45/2.”

The generator is factual because the event type comes from structured match data, not from the commentary examples.

### 10.8 `stream_demo.py`

This file demonstrates the full pipeline.

It performs the following process:

1. Load match events.
2. Build the commentary bank.
3. Loop over events in order.
4. Reset score when innings changes.
5. Classify each event.
6. Retrieve commentary examples.
7. Update score and wicket context.
8. Generate commentary.
9. Print event details and commentary.

## 11. Example End-to-End Flow

Example input:

```text
raw/1527575.json
```

Pipeline:

```text
Cricsheet JSON
    -> load_events.py
    -> flattened delivery dictionaries
    -> classify_event.py
    -> event labels
    -> load_commentary.py
    -> commentary bank
    -> retriever.py
    -> relevant commentary examples
    -> generator.py
    -> final commentary line
    -> stream_demo.py
    -> sequential match output
```

Example event path:

```text
Mohammad Rizwan faces Mohammad Ali
runs_off_bat = 4
extras = 0
wicket_type = ""
```

Classification:

```text
boundary_four
```

Possible generated commentary:

```text
That races to the fence. Four for Mohammad Rizwan with a crisp drive.
```

## 12. How to Run

From the project root, use:

```bash
python src/stream_demo.py
```

The current demo uses:

```python
run_match_demo("raw/1527574.json", "raw/train.csv", max_events=24)
```

To test individual pieces, you can run:

```bash
python src/test_loader.py
python src/test_retriever.py
python src/test_generator.py
```

## 13. Current Implementation Status

Completed:

- Fixed event label set.
- Cricsheet JSON flattening.
- Legal ball tracking for wides and no-balls.
- Event classification.
- Commentary CSV parsing.
- Commentary bank creation.
- TF-IDF vectorizer.
- Event-based commentary retrieval.
- Template-based generation.
- Style hints from retrieved commentary.
- Live score context in generated lines.
- Sequential match demo.

Still possible to improve:

- Add a true live sports API instead of replaying saved JSON.
- Add better evaluation metrics.
- Add support for more sports.
- Add a small LLM rewriting layer after the template generator works reliably.
- Fine-tune a model only after the event pipeline and retrieval baseline are stable.

## 14. Evaluation Plan

The best way to evaluate this project is not only classifier accuracy. Since the project is about commentary generation, evaluation should check both correctness and quality.

Useful evaluation questions:

1. **Event correctness:** Does the generated commentary match what happened on the ball?
2. **Factual consistency:** Does the commentary avoid inventing runs, wickets, or players?
3. **Fluency:** Does the generated line sound natural?
4. **Variety:** Does the system avoid repeating the same sentence too often?
5. **Context awareness:** Does the score context update correctly across the innings?
6. **Retrieval usefulness:** Do retrieved examples improve the wording style?

## 15. Known Limitations

The current project has some intentional limits:

- It is cricket-focused, not multi-sport.
- It uses saved match files instead of a live API.
- It uses templates and retrieval, not model fine-tuning.
- The commentary bank labels are rule-based, so some commentary rows may be grouped imperfectly.
- The generator produces short lines, not full broadcast-style paragraphs.
- The score context is simple and does not yet include advanced cricket statistics like required run rate, partnership, strike rate, or bowler figures.

These limitations are acceptable for the current scope because the project already demonstrates the key pipeline: structured sports data in, event classification, commentary retrieval, generated commentary out.

## 16. Future Work

Possible future improvements:

1. **Live Data Input**
   - Replace saved JSON replay with a live or periodically updated cricket feed.

2. **Better Context Features**
   - Add required run rate, current run rate, batter score, bowler figures, partnership, and match situation.

3. **Improved Retrieval**
   - Cache TF-IDF vectors instead of rebuilding them for every event.
   - Add better query features.
   - Use embeddings later if needed.

4. **LLM Polishing Layer**
   - Use a small language model to rewrite template output while preserving factual constraints.

5. **Fine-Tuning**
   - Fine-tune a model only after collecting enough high-quality event-commentary pairs.

6. **Multi-Sport Expansion**
   - Generalize the event label system to other sports after the cricket prototype is stable.

## 17. Summary

This project implements a working cricket commentary generator that converts ball-by-ball match data into short commentary lines. It uses structured data for factual correctness, a commentary corpus for realistic phrasing, TF-IDF retrieval for event-specific examples, and templates for controlled generation.

The current version is a realistic baseline for the assignment. It does not overreach into full fine-tuning or multi-sport support before the core pipeline is stable. That makes the project easier to explain, test, and demonstrate.
