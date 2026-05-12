from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data_io import find_column, infer_team_column


@dataclass(frozen=True)
class MatchSummary:
    match_id: str
    label: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_stats: dict[str, Any]
    away_stats: dict[str, Any]
    goals: pd.DataFrame


def match_options(frame: pd.DataFrame) -> list[str]:
    match_col = find_column(frame, ["match_id", "matchId", "game_id"])
    if match_col is None:
        return ["single match"]
    values = frame[match_col].dropna().astype(str).unique().tolist()
    return sorted(values)


def build_match_summary(frame: pd.DataFrame, match_id: str | None = None) -> MatchSummary:
    match_frame = _match_frame(frame, match_id)
    home_team, away_team = _resolve_home_away(match_frame)
    goals = _goal_events(match_frame)
    home_score, away_score = _resolve_score(match_frame, home_team, away_team, goals)
    resolved_match_id = _first_value(match_frame, ["match_id", "matchId", "game_id"]) or str(match_id or "single match")
    label = _first_value(match_frame, ["match_label", "label", "match"]) or f"{home_team} - {away_team}"

    return MatchSummary(
        match_id=str(resolved_match_id),
        label=str(label),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        home_stats=_team_stats(match_frame, home_team, home_score, away_score),
        away_stats=_team_stats(match_frame, away_team, away_score, home_score),
        goals=goals,
    )


def build_match_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id in match_options(frame):
        summary = build_match_summary(frame, match_id)
        rows.append(
            {
                "match_id": summary.match_id,
                "match": summary.label,
                "home_team": summary.home_team,
                "away_team": summary.away_team,
                "score": f"{summary.home_score}-{summary.away_score}",
                "home_shots": summary.home_stats["shots"],
                "away_shots": summary.away_stats["shots"],
                "home_shots_on_target": summary.home_stats["shots_on_target"],
                "away_shots_on_target": summary.away_stats["shots_on_target"],
                "goals": _goals_text(summary.goals),
            }
        )
    return pd.DataFrame(rows)


def build_team_match_outcomes(frame: pd.DataFrame, team_name: str) -> pd.DataFrame:
    rows = []
    for match_id in match_options(frame):
        summary = build_match_summary(frame, match_id)
        if team_name not in {summary.home_team, summary.away_team}:
            continue

        if team_name == summary.home_team:
            team_stats = summary.home_stats
            opp_stats = summary.away_stats
            opponent = summary.away_team
            venue = "Home"
            score_for = summary.home_score
            score_against = summary.away_score
            score = f"{summary.home_score}-{summary.away_score}"
        else:
            team_stats = summary.away_stats
            opp_stats = summary.home_stats
            opponent = summary.home_team
            venue = "Away"
            score_for = summary.away_score
            score_against = summary.home_score
            score = f"{summary.home_score}-{summary.away_score}"

        rows.append(
            {
                "include": True,
                "match_id": summary.match_id,
                "match": summary.label,
                "venue": venue,
                "opponent": opponent,
                "score": score,
                "result": _result(score_for, score_against),
                "shots": team_stats["shots"],
                "opponent_shots": opp_stats["shots"],
                "shots_on_target": team_stats["shots_on_target"],
                "opponent_shots_on_target": opp_stats["shots_on_target"],
                "pass_completion": team_stats["pass_completion"],
                "event_share": team_stats["event_share"],
                "goals": _goals_text(summary.goals[summary.goals["team_name"].astype(str) == team_name]),
            }
        )

    return pd.DataFrame(rows)


def format_match_summaries_for_prompt(outcomes: pd.DataFrame) -> str:
    if outcomes.empty:
        return "No match outcomes were selected."

    lines = ["Included match outcomes"]
    for row in outcomes.to_dict(orient="records"):
        lines.append(
            "- "
            + "; ".join(
                [
                    f"match={row.get('match')}",
                    f"venue={row.get('venue')}",
                    f"opponent={row.get('opponent')}",
                    f"score={row.get('score')}",
                    f"result={row.get('result')}",
                    f"shots={row.get('shots')}-{row.get('opponent_shots')}",
                    f"shots_on_target={row.get('shots_on_target')}-{row.get('opponent_shots_on_target')}",
                    f"pass_completion={row.get('pass_completion')}",
                    f"event_share={row.get('event_share')}",
                    f"goals={row.get('goals') or 'none'}",
                ]
            )
        )
    return "\n".join(lines)


def _match_frame(frame: pd.DataFrame, match_id: str | None) -> pd.DataFrame:
    match_col = find_column(frame, ["match_id", "matchId", "game_id"])
    if match_col is None or match_id is None or match_id == "single match":
        return frame.copy()
    return frame[frame[match_col].astype(str) == str(match_id)].copy()


def _resolve_home_away(frame: pd.DataFrame) -> tuple[str, str]:
    home = _first_value(frame, ["home_team", "home", "team_home"])
    away = _first_value(frame, ["away_team", "away", "team_away"])
    if home and away:
        return str(home), str(away)

    label = _first_value(frame, ["match_label", "label", "match"])
    if label:
        parsed = _parse_home_away_from_label(str(label))
        if parsed:
            return parsed

    team_col = infer_team_column(frame)
    if team_col:
        values = frame[team_col].dropna().astype(str).unique().tolist()
        if len(values) >= 2:
            return values[0], values[1]
        if len(values) == 1:
            return values[0], "Opponent"

    return "Home team", "Away team"


def _parse_home_away_from_label(label: str) -> tuple[str, str] | None:
    match = re.match(r"(.+?)\s+-\s+(.+?),\s+\d+", label)
    if not match:
        return None
    return match.group(1), match.group(2)


def _resolve_score(frame: pd.DataFrame, home_team: str, away_team: str, goals: pd.DataFrame) -> tuple[int, int]:
    home_score = _first_value(frame, ["home_score", "score_home", "home_goals"])
    away_score = _first_value(frame, ["away_score", "score_away", "away_goals"])
    if home_score is not None and away_score is not None:
        return _to_int(home_score), _to_int(away_score)

    return (
        int((goals["team_name"].astype(str) == home_team).sum()) if not goals.empty else 0,
        int((goals["team_name"].astype(str) == away_team).sum()) if not goals.empty else 0,
    )


def _team_stats(frame: pd.DataFrame, team_name: str, goals_for: int, goals_against: int) -> dict[str, Any]:
    team_col = infer_team_column(frame)
    team_frame = frame[frame[team_col].astype(str) == team_name] if team_col else frame.iloc[0:0]
    event_col = find_column(frame, ["event_name", "eventName", "event_type", "event"])
    sub_event_col = find_column(frame, ["sub_event_name", "subEventName", "sub_event", "action"])
    accurate_col = find_column(frame, ["accurate", "is_accurate", "success", "successful"])
    shot_on_target_col = find_column(frame, ["shot_on_target", "is_shot_on_target"])

    shots = _event_contains(team_frame, event_col, "shot") | _event_contains(team_frame, sub_event_col, "shot")
    passes = _event_contains(team_frame, event_col, "pass") | _event_contains(team_frame, sub_event_col, "pass")
    corners = _event_contains(team_frame, sub_event_col, "corner")
    fouls = _event_contains(team_frame, event_col, "foul") | _event_contains(team_frame, sub_event_col, "foul")
    accurate = _bool_series(team_frame[accurate_col]) if accurate_col else pd.Series(False, index=team_frame.index)
    goals = _goal_mask(team_frame)
    if shot_on_target_col:
        shots_on_target = _bool_series(team_frame[shot_on_target_col])
    else:
        shots_on_target = shots & (accurate | goals)
    pass_completion = _format_percent(_safe_rate(int((passes & accurate).sum()), int(passes.sum())))

    return {
        "team": team_name,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "events": len(team_frame),
        "event_share": _format_percent(_safe_rate(len(team_frame), len(frame))),
        "shots": int(shots.sum()),
        "shots_on_target": int(shots_on_target.sum()),
        "passes": int(passes.sum()),
        "pass_completion": pass_completion,
        "corners": int(corners.sum()),
        "fouls": int(fouls.sum()),
    }


def _goal_events(frame: pd.DataFrame) -> pd.DataFrame:
    team_col = infer_team_column(frame)
    player_col = find_column(frame, ["player_name", "player", "shortName"])
    period_col = find_column(frame, ["period", "matchPeriod"])
    second_col = find_column(frame, ["second", "eventSec"])
    goals = frame[_goal_mask(frame)].copy()

    if goals.empty:
        return pd.DataFrame(columns=["minute", "team_name", "player_name", "period", "second"])

    ordered_goals = goals.sort_values(by=[period_col, second_col]) if period_col and second_col else goals
    rows = []
    for _, row in ordered_goals.iterrows():
        rows.append(
            {
                "minute": _minute(row.get(period_col), row.get(second_col)) if period_col and second_col else "",
                "team_name": str(row.get(team_col, "")) if team_col else "",
                "player_name": str(row.get(player_col, "")) if player_col else "",
                "period": row.get(period_col, "") if period_col else "",
                "second": row.get(second_col, "") if second_col else "",
            }
        )
    return pd.DataFrame(rows)


def _goal_mask(frame: pd.DataFrame) -> pd.Series:
    goal_col = find_column(frame, ["goal", "is_goal"])
    if goal_col:
        raw_goal_mask = _bool_series(frame[goal_col])
    else:
        tag_col = find_column(frame, ["tag_ids", "tags"])
        if tag_col:
            raw_goal_mask = frame[tag_col].astype(str).str.contains(r"(^|\|)101(\||$)", regex=True, na=False)
        else:
            return pd.Series(False, index=frame.index)

    event_col = find_column(frame, ["event_name", "eventName", "event_type", "event"])
    sub_event_col = find_column(frame, ["sub_event_name", "subEventName", "sub_event", "action"])
    shot_mask = _event_contains(frame, event_col, "shot") | _event_contains(frame, sub_event_col, "shot")
    return raw_goal_mask & shot_mask


def _event_contains(frame: pd.DataFrame, column: str | None, text: str) -> pd.Series:
    if column is None:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.contains(text, case=False, na=False)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _first_value(frame: pd.DataFrame, aliases: list[str]) -> Any | None:
    column = find_column(frame, aliases)
    if column is None or frame.empty:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _minute(period: Any, second: Any) -> str:
    offsets = {"1H": 0, "2H": 45, "E1": 90, "E2": 105}
    try:
        raw_minute = int(math.floor(float(second) / 60)) + 1
    except (TypeError, ValueError):
        return ""
    return f"{offsets.get(str(period), 0) + raw_minute}'"


def _result(goals_for: int, goals_against: int) -> str:
    if goals_for > goals_against:
        return "Win"
    if goals_for < goals_against:
        return "Loss"
    return "Draw"


def _goals_text(goals: pd.DataFrame) -> str:
    if goals.empty:
        return ""
    parts = []
    for row in goals.to_dict(orient="records"):
        player = row.get("player_name") or "Unknown"
        minute = row.get("minute") or "?"
        team = row.get("team_name") or ""
        parts.append(f"{minute} {player} ({team})")
    return "; ".join(parts)
