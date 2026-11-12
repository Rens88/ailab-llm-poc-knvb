# AI Lab LLM Football Demo

This is a small Streamlit app for trying out a language model inside a sports-data app.

The app loads football match event data, builds a compact match or team profile, and asks an OpenAI model to describe a team's playing style. It is meant as a teaching demo for people who want to see how an app can call a language model through an API key.

The app does not send the full raw table to the model. It sends a smaller text summary that you can inspect in the app before making the model call.

## What You Need

- Python installed on your computer.
- An OpenAI API key.
- Internet access when you call the model.

The default model is `gpt-5-mini`.

## Install

Open a terminal in this project folder.

On macOS or Linux:

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

If `python` does not work on your machine, try `python3` instead.

## Start The App

```bash
streamlit run app.py
```

Streamlit will print a local web address, usually `http://localhost:8501`. Open that address in your browser.

## Add Your API Key

The easiest option is to paste your OpenAI API key into the **OpenAI API key** field in the app sidebar.

This key stays in the current browser/app session. It is not written to git.

If you do not add a key, you can still explore the sample data, profiles, and prompt preview. The buttons that call the model will stay disabled.

## Optional: Save A Local Key

If you do not want to paste your key each time, copy the example env file:

```bash
cp .env.example .env
```

Then edit `.env` and replace `YOUR_OPENAI_API_KEY` with your real key.

`.env` is ignored by git, so it should stay only on your machine.

## Use The Demo

The sidebar lets you choose between two pages:

- `LLM Match Characterization`: analyze one football match.
- `LLM MULTI-match Characterization`: analyze one team across several matches.

For the quickest demo, keep the sample data selected, choose a team, inspect the automatically generated profile, then click **Ask LLM**.

You can open **Model context** before calling the model to see exactly what text will be sent.

## Sample Data

This repository includes processed public Wyscout sample files under `data/sample/`.

If the sample files are missing, fetch them with:

```bash
python scripts/fetch_sample_data.py
```

The script downloads public football event data from `koenvo/wyscout-soccer-match-event-dataset` and writes app-ready CSV/JSON files into this project.

## Safety Notes

- Do not commit real API keys.
- Prefer pasting the key into the sidebar during workshops.
- Use `.env` only for a local machine you control.

## Sources

- Wyscout sample data: https://github.com/koenvo/wyscout-soccer-match-event-dataset
- OpenAI API key guidance: https://platform.openai.com/docs/api-reference/authentication
- OpenAI Responses API guidance: https://platform.openai.com/docs/guides/responses-vs-chat-completions
