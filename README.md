# Legal Sentence Game

A small offline CLI game for practicing legal clause drafting. It combines a built-in legal clause corpus, an interpolated n-gram word predictor, and a local clause classifier. No API key or network access is needed for the default experience.

## Setup

Use Python 3.12. Create and activate a virtual environment, then install the project requirements:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to configure a local classifier checkpoint or history path. The default classifier is a lightweight offline scorer. To use a locally fine-tuned GPT-2 sequence classifier, install compatible `torch` and `transformers` packages separately and point `GPT2_MODEL_PATH` to its directory; the application loads it with local files only.

## Play

```powershell
python -m app.cli hello
python -m app.cli play
```

The game shows a target clause type and starter. Choose one of the three predicted next tokens, type custom text, use `/undo`, or enter `.` / `/done` to classify and save the sentence. Use `python -m app.cli play --help` for `--rounds` and `--data` options. Session history is stored in `.data/history.json` by default.

## Tests

```powershell
python -m pytest -v
```
