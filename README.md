# Legal Sentence Game

A small offline CLI game for practicing legal clause drafting. It combines a built-in legal clause corpus, an interpolated n-gram word predictor, and a local clause classifier. No API key or network access is needed for the default experience.

## Setup

Use Python 3.12. Create and activate a virtual environment, then install the project requirements:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt