# AI Lab LLM Tabular Data Demo

This repository demonstrates a Streamlit app where data engineers can let an LLM inspect tabular football event data without handing it an entire raw dataset. The app loads CSV or JSON event tables, builds auditable team-level profiles, and asks OpenAI to characterize a selected team's playing style. Optional text notes can be added as unstructured context.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Credentials

Create an API key in the OpenAI Platform dashboard. Keep the real key out of git.

Option 1: environment variable:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export OPENAI_MODEL="gpt-5.4-mini"
```

Option 2: local Streamlit secrets:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`. That file is ignored by git.

## App Pages

- `LLM Match Characterization`: analyze one selected match, review home/away score, shots, shots on target, pass completion, corners, fouls, and goal scorers, then ask the LLM for a characterization.
- `LLM MULTI-match Characterization`: load a collection of match files, select a team, include or exclude specific matches with checkboxes, review each match outcome, then ask the LLM for a multi-match characterization.

Both pages include a `Steer` field for generating a second characterization with a specific emphasis or clarification.

## Sample Data

Fetch the included public Wyscout samples:

```bash
python scripts/fetch_sample_data.py
```

The script downloads seven processed Manchester City matches from `koenvo/wyscout-soccer-match-event-dataset`. It keeps one original nested Wyscout JSON under `data/raw/`, writes individual app-ready match CSV files under `data/sample/matches/`, and writes single-match plus multi-match CSV/JSON examples under `data/sample/`.

## Run

```bash
streamlit run app.py
```

The app sends a compact table profile to the OpenAI Responses API, not the full raw table. You can inspect the exact model context before calling the LLM.

## Sources

- Wyscout sample data: https://github.com/koenvo/wyscout-soccer-match-event-dataset
- OpenAI API key guidance: https://platform.openai.com/docs/api-reference/authentication
- OpenAI Responses API guidance: https://platform.openai.com/docs/guides/responses-vs-chat-completions
