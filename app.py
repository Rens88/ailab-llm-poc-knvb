from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from football_llm_demo.data_io import (
    infer_team_column,
    load_table_from_path,
    load_table_from_upload,
    team_values,
)
from football_llm_demo.llm import build_style_prompt, generate_style_analysis
from football_llm_demo.summarizer import build_team_profile


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "data" / "sample"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def main() -> None:
    st.set_page_config(page_title="Football LLM Table Explorer", layout="wide")
    st.title("Football LLM Table Explorer")

    api_key, key_source = _resolve_api_key()

    with st.sidebar:
        st.header("OpenAI")
        if api_key:
            st.success(f"Key loaded from {key_source}.")
        else:
            st.warning("No API key found.")
        model = st.text_input("Model", value=_resolve_model())

        st.header("Data")
        source_mode = st.radio("Table source", ["Sample file", "Upload file"], horizontal=False)
        loaded_table = _load_table(source_mode)

    if loaded_table is None:
        st.info("Add a sample file or upload a CSV/JSON table.")
        return

    frame = loaded_table.dataframe
    st.caption(f"{loaded_table.name} - {loaded_table.source_kind} - {len(frame):,} rows x {len(frame.columns):,} columns")

    left, right = st.columns([0.35, 0.65])
    with left:
        team_column = _team_column_picker(frame)
        if not team_column:
            st.error("No team column was detected. Upload a table with a team column.")
            return
        values = team_values(frame, team_column)
        if not values:
            st.error("The selected team column has no values.")
            return
        default_index = _default_team_index(values)
        selected_team = st.selectbox("Team", values, index=default_index)
        question = st.text_area(
            "Question",
            value=f"What characterizes the playing style of {selected_team}?",
            height=90,
        )
        notes = _unstructured_context()

    with right:
        profile = build_team_profile(frame, selected_team, team_column)
        _render_profile(profile)

    prompt = build_style_prompt(
        team_name=selected_team,
        table_context=profile.prompt_context,
        analyst_question=question,
        unstructured_context=notes,
    )

    with st.expander("Model context"):
        st.code(prompt, language="text")

    if st.button("Ask LLM", type="primary", disabled=not bool(api_key)):
        with st.spinner("Generating analysis"):
            try:
                answer = generate_style_analysis(api_key=api_key or "", model=model.strip(), prompt=prompt)
            except Exception as exc:
                st.error(f"OpenAI request failed: {exc}")
            else:
                st.markdown(answer)

    if not api_key:
        st.info("Set OPENAI_API_KEY or .streamlit/secrets.toml to enable the model call. The profile and prompt preview still run locally.")

    with st.expander("Table preview"):
        st.dataframe(frame.head(250), width='stretch')


def _load_table(source_mode: str):
    if source_mode == "Upload file":
        uploaded = st.file_uploader("CSV or JSON table", type=["csv", "json"])
        if uploaded is None:
            return None
        return load_table_from_upload(uploaded, uploaded.name)

    sample_files = sorted(SAMPLE_DIR.glob("*.csv")) + sorted(SAMPLE_DIR.glob("*.json"))
    if not sample_files:
        st.warning("No sample files found. Run python scripts/fetch_sample_data.py.")
        return None
    selected = st.selectbox("Sample table", sample_files, format_func=lambda path: path.name)
    return load_table_from_path(selected)


def _team_column_picker(frame: pd.DataFrame) -> str | None:
    inferred = infer_team_column(frame)
    options = list(frame.columns)
    if inferred and inferred in options:
        index = options.index(inferred)
    else:
        index = 0
    return st.selectbox("Team column", options, index=index)


def _default_team_index(values: list[str]) -> int:
    preferred = ["Manchester City", "Liverpool"]
    for team in preferred:
        if team in values:
            return values.index(team)
    return 0


def _unstructured_context() -> str:
    sample_notes = SAMPLE_DIR / "wyscout_2499754_notes.md"
    if sample_notes.exists():
        use_sample_notes = st.checkbox("Use sample notes", value=False)
    else:
        use_sample_notes = False
    uploaded_notes = st.file_uploader("Optional notes", type=["txt", "md"])
    pasted_notes = st.text_area("Optional pasted notes", height=120)
    chunks: list[str] = []
    if use_sample_notes:
        chunks.append(sample_notes.read_text(encoding="utf-8"))
    if uploaded_notes is not None:
        chunks.append(uploaded_notes.getvalue().decode("utf-8", errors="replace"))
    if pasted_notes.strip():
        chunks.append(pasted_notes.strip())
    return "\n\n".join(chunks)


def _render_profile(profile) -> None:
    metric_frame = pd.DataFrame(profile.metrics)
    display_columns = ["metric", "selected_display", "reference_display"]
    st.subheader("Team Profile")
    st.dataframe(
        metric_frame[display_columns].rename(
            columns={
                "metric": "Metric",
                "selected_display": profile.team_name,
                "reference_display": "Other rows",
            }
        ),
        hide_index=True,
        width='stretch',
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Event Mix")
        st.dataframe(profile.top_events, hide_index=True, width='stretch')
    with col_b:
        st.subheader("Sub-Events")
        st.dataframe(profile.top_sub_events, hide_index=True, width='stretch')


def _resolve_api_key() -> tuple[str | None, str | None]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key, "OPENAI_API_KEY"

    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]), ".streamlit/secrets.toml"
        openai_section = st.secrets.get("openai", {})
        if isinstance(openai_section, dict) and openai_section.get("api_key"):
            return str(openai_section["api_key"]), ".streamlit/secrets.toml"
    except Exception:
        pass

    return None, None


def _resolve_model() -> str:
    try:
        if "OPENAI_MODEL" in st.secrets:
            return str(st.secrets["OPENAI_MODEL"])
    except Exception:
        pass
    return DEFAULT_MODEL


if __name__ == "__main__":
    main()
