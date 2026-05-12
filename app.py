from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from football_llm_demo.data_io import (
    find_column,
    infer_team_column,
    load_table_from_path,
    load_table_from_upload,
    team_values,
)
from football_llm_demo.llm import build_style_prompt, generate_style_analysis
from football_llm_demo.match_stats import (
    build_match_summary,
    build_team_match_outcomes,
    format_match_summaries_for_prompt,
    match_options,
)
from football_llm_demo.summarizer import build_team_profile


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "data" / "sample"
MATCH_SAMPLE_DIR = SAMPLE_DIR / "matches"
ASSET_DIR = ROOT / "assets"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
PAGES = ["LLM Match Characterization", "LLM MULTI-match Characterization"]


def main() -> None:
    st.set_page_config(page_title="LLM Match Characterization", layout="wide")
    _install_styles()
    _render_main_brand()

    api_key, key_source = _resolve_api_key()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Page", PAGES, label_visibility="collapsed")

        st.header("OpenAI")
        if api_key:
            st.success(f"Key loaded from {key_source}.")
        else:
            st.warning("No API key found.")
        model = st.text_input("Model", value=_resolve_model())
        _render_brand_footer()

    if page == "LLM MULTI-match Characterization":
        _page_multi_match(api_key, model)
    else:
        _page_single_match(api_key, model)


def _page_single_match(api_key: str | None, model: str) -> None:
    st.title("LLM Match Characterization")
    st.caption("Analyze one football match from a CSV or JSON event table.")

    with st.sidebar:
        st.header("Single-match data")
        source_mode = st.radio("Table source", ["Sample file", "Upload file"], horizontal=False, key="single_source")
        loaded_table = _load_table(source_mode, key_prefix="single")

    if loaded_table is None:
        st.info("Add a sample file or upload a CSV/JSON table.")
        return

    frame = loaded_table.dataframe
    selected_match = _match_picker(frame, key="single_match")
    match_frame = _filter_match(frame, selected_match)
    summary = build_match_summary(match_frame, selected_match)
    st.caption(
        f"{loaded_table.name} - {loaded_table.source_kind} - "
        f"{len(match_frame):,} rows x {len(match_frame.columns):,} columns"
    )
    _render_match_summary(summary)

    left, right = st.columns([0.35, 0.65])
    with left:
        team_column = _team_column_picker(match_frame, key="single_team_column")
        if not team_column:
            st.error("No team column was detected. Upload a table with a team column.")
            return
        values = team_values(match_frame, team_column)
        if not values:
            st.error("The selected team column has no values.")
            return
        default_index = _default_team_index(values)
        selected_team = st.selectbox("Team", values, index=default_index, key="single_team")
        question = st.text_area(
            "Question",
            value=f"What characterizes the playing style of {selected_team}?",
            height=90,
            key="single_question",
        )
        notes = _unstructured_context(key_prefix="single")

    with right:
        profile = build_team_profile(match_frame, selected_team, team_column)
        _render_profile(profile)

    match_context = _match_context_for_prompt(summary)
    table_context = f"{profile.prompt_context}\n\n{match_context}"
    _llm_controls(
        api_key=api_key,
        model=model,
        selected_team=selected_team,
        table_context=table_context,
        question=question,
        notes=notes,
        key_prefix="single",
    )

    with st.expander("Table preview"):
        st.dataframe(match_frame.head(250), width='stretch')


def _page_multi_match(api_key: str | None, model: str) -> None:
    st.title("LLM MULTI-match Characterization")
    st.caption("Analyze a selected team across a collection of match files.")

    with st.sidebar:
        st.header("Multi-match data")
        source_mode = st.radio(
            "Collection source",
            ["Sample collection", "Upload collection"],
            horizontal=False,
            key="multi_source",
        )
        loaded_tables = _load_collection(source_mode)

    if not loaded_tables:
        st.info("Add sample match files or upload a collection of CSV/JSON files.")
        return

    frame = _combine_tables(loaded_tables)
    st.caption(
        f"{len(loaded_tables):,} files - {len(frame):,} rows x {len(frame.columns):,} columns"
    )

    team_column = infer_team_column(frame)
    if team_column is None:
        st.error("No team column was detected. Upload files with a team column.")
        return

    values = team_values(frame, team_column)
    selected_team = st.selectbox(
        "Team",
        values,
        index=_default_team_index(values),
        key="multi_team",
    )

    outcomes = build_team_match_outcomes(frame, selected_team)
    if outcomes.empty:
        st.warning("No matches were found for that team.")
        return

    st.subheader("Loaded Match Outcomes")
    edited_outcomes = st.data_editor(
        outcomes,
        column_config={
            "include": st.column_config.CheckboxColumn("Include", default=True),
        },
        disabled=[column for column in outcomes.columns if column != "include"],
        hide_index=True,
        width='stretch',
        key=f"multi_match_selector_{selected_team}",
    )
    included = edited_outcomes[edited_outcomes["include"]]
    if included.empty:
        st.warning("Select at least one match to analyze.")
        return

    included_ids = set(included["match_id"].astype(str))
    match_col = _match_column(frame)
    selected_frame = frame[frame[match_col].astype(str).isin(included_ids)] if match_col else frame

    left, right = st.columns([0.35, 0.65])
    with left:
        question = st.text_area(
            "Question",
            value=f"What characterizes the playing style of {selected_team} across these matches?",
            height=90,
            key="multi_question",
        )
        notes = _unstructured_context(key_prefix="multi")

    with right:
        profile = build_team_profile(selected_frame, selected_team, team_column)
        _render_profile(profile)

    outcome_context = format_match_summaries_for_prompt(included.drop(columns=["include"]))
    table_context = f"{profile.prompt_context}\n\n{outcome_context}"
    _llm_controls(
        api_key=api_key,
        model=model,
        selected_team=selected_team,
        table_context=table_context,
        question=question,
        notes=notes,
        key_prefix="multi",
    )

    with st.expander("Combined table preview"):
        st.dataframe(selected_frame.head(500), width='stretch')


def _load_table(source_mode: str, key_prefix: str):
    if source_mode == "Upload file":
        uploaded = st.file_uploader("CSV or JSON table", type=["csv", "json"], key=f"{key_prefix}_upload")
        if uploaded is None:
            return None
        return load_table_from_upload(uploaded, uploaded.name)

    sample_files = sorted(SAMPLE_DIR.glob("*.csv")) + sorted(SAMPLE_DIR.glob("*.json"))
    sample_files = [path for path in sample_files if path.name != "wyscout_multi_match_events.json"]
    sample_files.sort(key=lambda path: ("2499754" not in path.name, "multi" in path.name, path.name))
    if not sample_files:
        st.warning("No sample files found. Run python scripts/fetch_sample_data.py.")
        return None
    selected = st.selectbox(
        "Sample table",
        sample_files,
        format_func=lambda path: path.name,
        key=f"{key_prefix}_sample_table",
    )
    return load_table_from_path(selected)


def _load_collection(source_mode: str):
    if source_mode == "Upload collection":
        uploaded_files = st.file_uploader(
            "CSV or JSON match files",
            type=["csv", "json"],
            accept_multiple_files=True,
            key="multi_uploads",
        )
        return [load_table_from_upload(uploaded, uploaded.name) for uploaded in uploaded_files]

    sample_files = sorted(MATCH_SAMPLE_DIR.glob("*.csv"))
    if not sample_files:
        fallback = SAMPLE_DIR / "wyscout_multi_match_events.csv"
        sample_files = [fallback] if fallback.exists() else []
    if not sample_files:
        st.warning("No sample match collection found. Run python scripts/fetch_sample_data.py.")
        return []

    selected_files = st.multiselect(
        "Sample match files",
        sample_files,
        default=sample_files,
        format_func=lambda path: path.name,
        key="multi_sample_files",
    )
    return [load_table_from_path(path) for path in selected_files]


def _combine_tables(loaded_tables) -> pd.DataFrame:
    frames = []
    for table in loaded_tables:
        frame = table.dataframe.copy()
        frame["source_file"] = table.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _match_picker(frame: pd.DataFrame, key: str) -> str | None:
    matches = match_options(frame)
    if len(matches) <= 1:
        return matches[0] if matches else None
    return st.selectbox("Match", matches, key=key)


def _filter_match(frame: pd.DataFrame, match_id: str | None) -> pd.DataFrame:
    match_col = _match_column(frame)
    if match_col is None or match_id is None or match_id == "single match":
        return frame
    return frame[frame[match_col].astype(str) == str(match_id)].copy()


def _match_column(frame: pd.DataFrame) -> str | None:
    return find_column(frame, ["match_id", "matchId", "game_id"])


def _team_column_picker(frame: pd.DataFrame, key: str) -> str | None:
    inferred = infer_team_column(frame)
    options = list(frame.columns)
    if inferred and inferred in options:
        index = options.index(inferred)
    else:
        index = 0
    return st.selectbox("Team column", options, index=index, key=key)


def _default_team_index(values: list[str]) -> int:
    preferred = ["Manchester City", "Liverpool"]
    for team in preferred:
        if team in values:
            return values.index(team)
    return 0


def _unstructured_context(key_prefix: str) -> str:
    sample_notes = SAMPLE_DIR / "wyscout_2499754_notes.md"
    if sample_notes.exists():
        use_sample_notes = st.checkbox("Use sample notes", value=False, key=f"{key_prefix}_sample_notes")
    else:
        use_sample_notes = False
    uploaded_notes = st.file_uploader("Optional notes", type=["txt", "md"], key=f"{key_prefix}_notes_upload")
    pasted_notes = st.text_area("Optional pasted notes", height=120, key=f"{key_prefix}_pasted_notes")
    chunks: list[str] = []
    if use_sample_notes:
        chunks.append(sample_notes.read_text(encoding="utf-8"))
    if uploaded_notes is not None:
        chunks.append(uploaded_notes.getvalue().decode("utf-8", errors="replace"))
    if pasted_notes.strip():
        chunks.append(pasted_notes.strip())
    return "\n\n".join(chunks)


def _render_match_summary(summary) -> None:
    st.subheader("Match Summary")
    home, score, away = st.columns([0.42, 0.16, 0.42])
    with home:
        _team_match_card("Home", summary.home_stats)
    with score:
        st.markdown(
            f"""
            <div class="score-card">
              <div class="score-label">Score</div>
              <div class="score-value">{summary.home_score} - {summary.away_score}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with away:
        _team_match_card("Away", summary.away_stats)

    if not summary.goals.empty:
        st.dataframe(
            summary.goals[["minute", "team_name", "player_name"]].rename(
                columns={"minute": "Minute", "team_name": "Team", "player_name": "Scorer"}
            ),
            hide_index=True,
            width='stretch',
        )
    else:
        st.caption("No goals found in the event data.")


def _team_match_card(label: str, stats: dict) -> None:
    st.markdown(
        f"""
        <div class="match-card">
          <div class="match-side">{label}</div>
          <div class="match-team">{stats["team"]}</div>
          <div class="match-grid">
            <div><span>Goals</span><strong>{stats["goals_for"]}</strong></div>
            <div><span>Shots</span><strong>{stats["shots"]}</strong></div>
            <div><span>On target</span><strong>{stats["shots_on_target"]}</strong></div>
            <div><span>Passes</span><strong>{stats["passes"]}</strong></div>
            <div><span>Pass comp.</span><strong>{stats["pass_completion"]}</strong></div>
            <div><span>Events</span><strong>{stats["event_share"]}</strong></div>
            <div><span>Corners</span><strong>{stats["corners"]}</strong></div>
            <div><span>Fouls</span><strong>{stats["fouls"]}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _llm_controls(
    api_key: str | None,
    model: str,
    selected_team: str,
    table_context: str,
    question: str,
    notes: str,
    key_prefix: str,
) -> None:
    prompt = build_style_prompt(
        team_name=selected_team,
        table_context=table_context,
        analyst_question=question,
        unstructured_context=notes,
    )

    with st.expander("Model context"):
        st.code(prompt, language="text")

    if st.button("Ask LLM", type="primary", disabled=not bool(api_key), key=f"{key_prefix}_ask"):
        _run_llm(api_key, model, prompt)

    steering = st.text_area(
        "Steer",
        placeholder="Suggest a direction, emphasis, or clarification for a second characterization.",
        height=90,
        key=f"{key_prefix}_steer",
    )
    steered_prompt = build_style_prompt(
        team_name=selected_team,
        table_context=table_context,
        analyst_question=question,
        unstructured_context=notes,
        steering=steering,
    )
    if st.button(
        "Generate steered characterization",
        disabled=not bool(api_key) or not bool(steering.strip()),
        key=f"{key_prefix}_ask_steered",
    ):
        _run_llm(api_key, model, steered_prompt)

    if not api_key:
        st.info("Set OPENAI_API_KEY or .streamlit/secrets.toml to enable model calls. The profile and prompt preview still run locally.")


def _run_llm(api_key: str | None, model: str, prompt: str) -> None:
    with st.spinner("Generating analysis"):
        try:
            answer = generate_style_analysis(api_key=api_key or "", model=model.strip(), prompt=prompt)
        except Exception as exc:
            st.error(f"OpenAI request failed: {exc}")
        else:
            st.markdown(answer)


def _match_context_for_prompt(summary) -> str:
    if summary.goals.empty:
        goals = "No goals found in the event rows."
    else:
        goals = "; ".join(
            f"{row['minute']} {row['player_name']} ({row['team_name']})"
            for row in summary.goals.to_dict(orient="records")
        )
    return "\n".join(
        [
            "Match score context",
            f"- Match: {summary.label}",
            f"- Home: {summary.home_team}",
            f"- Away: {summary.away_team}",
            f"- Score: {summary.home_score}-{summary.away_score}",
            f"- Goals: {goals}",
        ]
    )


def _install_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
        }
        h1, h2, h3 {
            color: #162033;
            letter-spacing: 0;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #dde3ee;
        }
        .match-card, .score-card {
            background: #ffffff;
            border: 1px solid #dce3ed;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 178px;
            box-shadow: 0 1px 2px rgba(22, 32, 51, 0.05);
        }
        .score-card {
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .score-label, .match-side {
            color: #657187;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .score-value {
            color: #162033;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .match-team {
            color: #162033;
            font-size: 1.2rem;
            font-weight: 750;
            margin: 4px 0 12px;
        }
        .match-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
        }
        .match-grid div {
            background: #f1f5f9;
            border-radius: 6px;
            padding: 8px;
        }
        .match-grid span {
            color: #657187;
            display: block;
            font-size: 0.72rem;
            line-height: 1.15;
        }
        .match-grid strong {
            color: #162033;
            display: block;
            font-size: 0.95rem;
            margin-top: 2px;
        }
        .brand-footer {
            position: fixed;
            left: 1.2rem;
            bottom: 0.8rem;
            width: 16rem;
            background: rgba(255, 255, 255, 0.94);
            border-top: 1px solid #dde3ee;
            padding-top: 0.65rem;
            z-index: 99;
        }
        .brand-logos {
            display: flex;
            gap: 0.55rem;
            align-items: center;
            margin-bottom: 0.35rem;
        }
        .brand-logos img {
            max-height: 30px;
            max-width: 74px;
            object-fit: contain;
        }
        .brand-subtitle {
            color: #526174;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        .main-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.15rem 0 0.75rem;
        }
        .main-brand img {
            max-height: 38px;
            max-width: 100px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_main_brand() -> None:
    image_tags = _logo_image_tags(max_count=3)
    if not image_tags:
        return

    st.markdown(
        f'<div class="main-brand">{"".join(image_tags)}</div>',
        unsafe_allow_html=True,
    )


def _render_brand_footer() -> None:
    image_tags = _logo_image_tags(max_count=3)
    if not image_tags:
        return

    st.markdown(
        f"""
        <div class="brand-footer">
          <div class="brand-logos">{"".join(image_tags)}</div>
          <div class="brand-subtitle">PoC-app generated for AI Impact Labs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _logo_image_tags(max_count: int) -> list[str]:
    logos = [
        ASSET_DIR / "revelx-logo.png",
        ASSET_DIR / "dsa-logo.png",
        ASSET_DIR / "sportinnovator-ai-lab-logo.png",
    ]
    image_tags = []
    for logo in logos[:max_count]:
        if logo.exists():
            image_tags.append(f'<img src="{_image_data_uri(logo)}" alt="{logo.stem}">')
    return image_tags


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
