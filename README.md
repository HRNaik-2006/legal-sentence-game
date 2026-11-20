# Legal Sentence Game

An offline CLI for practicing short legal-clause drafting. It suggests next tokens with an interpolated trigram model, classifies the finished clause, and can score its predictor against a simple unigram baseline.

## Requirements and setup

Python 3.11 or newer. The game and fallback classifier use the Python standard library; [requirements.txt](./requirements.txt) pins pytest and its test-runner dependencies. Optional local Transformers checkpoint support is separate; model files are not downloaded.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If pip is missing from the environment, run `python -m ensurepip --upgrade` before installing requirements. On Linux or macOS, activate with `source .venv/bin/activate`.

## Run

```powershell
python -m app.cli hello
python -m app.cli play
python -m app.cli play --rounds 1 --data .data/history.json
python -m app.cli evaluate
```

During play, choose one of the suggestions, enter custom text, use `/undo`, or enter `.` / `/done` to finish a clause. `play --help` lists its options. Completed rounds are stored under `.data/history.json` unless `--data` or `LEGAL_GAME_DATA` overrides the path.

## Data source

The default corpus is the `LEGAL_CLAUSES` mapping in [app/core.py](./app/core.py): 64 short examples across eight clause categories. They are bundled directly in the source; the application does not fetch a corpus at runtime. The repository does not record an external source or authorship for these example strings, so treat them as illustrative, non-authoritative text and verify provenance before reusing them in a published or commercial dataset.

## Results and sample output

Captured with Python 3.11.15 and pytest 9.2.0 (output excerpt):

```text
$ python -m pytest -v
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0
collected 12 items

tests/test_core.py::test_builtin_corpus_and_sentence_boundaries PASSED
tests/test_core.py::test_ngram_predictions_are_top_three_and_normalized PASSED
tests/test_core.py::test_evaluation_compares_model_and_unigram_on_repeatable_holdout PASSED
tests/test_core.py::test_evaluation_rejects_empty_or_too_small_corpora PASSED
tests/test_core.py::test_classifier_schema_and_category_recognition PASSED
tests/test_core.py::test_storage_round_trip_and_progress PASSED
tests/test_core.py::test_storage_surfaces_corrupt_history_without_overwriting PASSED
tests/test_core.py::test_storage_rejects_unexpected_history_shape PASSED
tests/test_core.py::test_cli_game_records_round PASSED
tests/test_smoke.py::test_hello_command PASSED
tests/test_smoke.py::test_help_command PASSED
tests/test_smoke.py::test_evaluate_command_reports_baseline PASSED
============================== 12 passed in 0.94s ==============================
```

The evaluator uses every fourth built-in clause as a deterministic holdout and scores each next token given the preceding gold tokens:

```text
$ python -m app.cli evaluate
Built-in clauses: 64 (48 train, 16 held out)
Next tokens scored: 228
Interpolated trigram: top-1 21.9%, top-3 35.1%
Unigram baseline: top-1 7.0%, top-3 17.5%
```

A one-round CLI session can be replayed without typing interactively in PowerShell:

```powershell
'.' | python -m app.cli play --rounds 1 --data .tmp\sample-history.json
```

Sample captured output (the target and suggestions vary between runs):

```text
Target clause: Termination
Starter: Either party may terminate this agreement
Draft: Either party may terminate this agreement
[1] . (55.3%)
[2] shall (22.9%)
[3] is (21.8%)
Choose 1/2/3, type a word or phrase, /undo, or . to finish:
Sentence: Either party may terminate this agreement.
Predicted clause: Termination (86.5%)
Target match: Yes
Feedback: Specify notice, cure periods, termination triggers, and post-termination duties. Add enough detail to make the obligation and its limits clear.
```

## Baseline and limitations

The built-in evaluation compares the trigram predictor with a unigram baseline ranked by training-token frequency. On the captured 48-clause training / 16-clause holdout split, top-1 accuracy was 21.9% versus 7.0%; top-3 accuracy was 35.1% versus 17.5%. These are internal results on a small, in-domain sample, not an independent benchmark; the split is deterministic and clauses may share wording. Punctuation and sentence-end tokens are included in the scored next-token sequence.

The trigram model conditions on at most two preceding tokens, cannot represent legal meaning or long-range clause structure, and may prefer memorized phrases. Its displayed probabilities are normalized over the returned suggestions, not calibrated confidence across all possible text. The default classifier uses keyword hits and corpus-word overlap rather than legal reasoning; its confidence is not calibrated. An optional local Transformers checkpoint can be used when present, but neither model output is legal advice.

## Tests

CI installs [requirements.txt](./requirements.txt) and runs the same command:

```powershell
python -m pytest -v
```
