from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data_io import find_column, infer_team_column


@dataclass(frozen=True)
class TeamProfile:
    team_name: str
    team_column: str
    total_rows: int
    selected_rows: int
    metrics: list[dict[str, Any]]
    top_events: pd.DataFrame
    top_sub_events: pd.DataFrame
    prompt_context: str


def build_team_profile(frame: pd.DataFrame, team_name: str, team_column: str | None = None) -> TeamProfile:
    resolved_team_column = team_column or infer_team_column(frame)
    if resolved_team_column is None:
        raise ValueError("No team column found. Choose a column that identifies teams.")

    selected = frame[frame[resolved_team_column].astype(str) == str(team_name)].copy()
    if selected.empty:
        raise ValueError(f"No rows found for team '{team_name}' in column '{resolved_team_column}'.")

    other = frame[frame[resolved_team_column].astype(str) != str(team_name)].copy()
    metrics = _football_metrics(selected, other, frame)
    top_events = _top_values(selected, _event_column(frame), "event")
    top_sub_events = _top_values(selected, _sub_event_column(frame), "sub_event")
    prompt_context = _format_prompt_context(
        frame=frame,
        selected=selected,
        other=other,
        team_name=team_name,
        team_column=resolved_team_column,
        metrics=metrics,
        top_events=top_events,
        top_sub_events=top_sub_events,
    )

    return TeamProfile(
        team_name=str(team_name),
        team_column=resolved_team_column,
        total_rows=len(frame),
        selected_rows=len(selected),
        metrics=metrics,
        top_events=top_events,
        top_sub_events=top_sub_events,
        prompt_context=prompt_context,
    )


def _football_metrics(selected: pd.DataFrame, other: pd.DataFrame, frame: pd.DataFrame) -> list[dict[str, Any]]:
    event_col = _event_column(frame)
    sub_event_col = _sub_event_column(frame)
    accurate_col = find_column(frame, ["accurate", "is_accurate", "success", "successful"])
    start_x_col = find_column(frame, ["start_x", "x", "location_x"])
    start_y_col = find_column(frame, ["start_y", "y", "location_y"])
    end_x_col = find_column(frame, ["end_x", "end_location_x"])

    def mask_contains(data: pd.DataFrame, column: str | None, text: str) -> pd.Series:
        if column is None:
            return pd.Series(False, index=data.index)
        return data[column].astype(str).str.contains(text, case=False, na=False)

    pass_mask = mask_contains(selected, event_col, "pass") | mask_contains(selected, sub_event_col, "pass")
    pass_mask_other = mask_contains(other, event_col, "pass") | mask_contains(other, sub_event_col, "pass")
    shot_mask = mask_contains(selected, event_col, "shot") | mask_contains(selected, sub_event_col, "shot")
    shot_mask_other = mask_contains(other, event_col, "shot") | mask_contains(other, sub_event_col, "shot")
    duel_mask = mask_contains(selected, event_col, "duel") | mask_contains(selected, sub_event_col, "duel")
    duel_mask_other = mask_contains(other, event_col, "duel") | mask_contains(other, sub_event_col, "duel")
    cross_mask = mask_contains(selected, sub_event_col, "cross")
    cross_mask_other = mask_contains(other, sub_event_col, "cross")
    high_pass_mask = mask_contains(selected, sub_event_col, "high pass")
    high_pass_mask_other = mask_contains(other, sub_event_col, "high pass")
    set_piece_mask = mask_contains(selected, event_col, "free kick") | mask_contains(selected, sub_event_col, "corner")
    set_piece_mask_other = mask_contains(other, event_col, "free kick") | mask_contains(other, sub_event_col, "corner")

    metrics: list[dict[str, Any]] = [
        _count_metric("Rows in table", len(selected), len(other)),
        _rate_metric("Event share of table", len(selected), len(frame), len(other), len(frame)),
        _rate_metric("Pass share", int(pass_mask.sum()), len(selected), int(pass_mask_other.sum()), len(other)),
        _rate_metric("Shot share", int(shot_mask.sum()), len(selected), int(shot_mask_other.sum()), len(other)),
        _rate_metric("Duel share", int(duel_mask.sum()), len(selected), int(duel_mask_other.sum()), len(other)),
        _rate_metric("Cross share", int(cross_mask.sum()), len(selected), int(cross_mask_other.sum()), len(other)),
        _rate_metric("High pass share", int(high_pass_mask.sum()), len(selected), int(high_pass_mask_other.sum()), len(other)),
        _rate_metric("Set-piece share", int(set_piece_mask.sum()), len(selected), int(set_piece_mask_other.sum()), len(other)),
    ]

    if accurate_col:
        selected_passes = selected[pass_mask]
        other_passes = other[pass_mask_other]
        metrics.append(
            _rate_metric(
                "Pass completion",
                int(selected_passes[accurate_col].fillna(False).astype(bool).sum()),
                len(selected_passes),
                int(other_passes[accurate_col].fillna(False).astype(bool).sum()),
                len(other_passes),
            )
        )

    if start_x_col:
        metrics.append(
            _mean_metric("Average start x", selected[start_x_col], other[start_x_col] if not other.empty else None)
        )
        metrics.append(
            _rate_metric(
                "Final-third starts",
                int(pd.to_numeric(selected[start_x_col], errors="coerce").ge(67).sum()),
                len(selected),
                int(pd.to_numeric(other[start_x_col], errors="coerce").ge(67).sum()) if not other.empty else 0,
                len(other),
            )
        )

    if start_y_col:
        metrics.append(
            _mean_metric(
                "Average width from center",
                (pd.to_numeric(selected[start_y_col], errors="coerce") - 50).abs(),
                (pd.to_numeric(other[start_y_col], errors="coerce") - 50).abs() if not other.empty else None,
            )
        )

    if start_x_col and end_x_col:
        selected_progressive = (
            pd.to_numeric(selected[end_x_col], errors="coerce")
            - pd.to_numeric(selected[start_x_col], errors="coerce")
        ).ge(25)
        other_progressive = (
            pd.to_numeric(other[end_x_col], errors="coerce")
            - pd.to_numeric(other[start_x_col], errors="coerce")
        ).ge(25)
        metrics.append(
            _rate_metric(
                "Progressive-pass share",
                int((selected_progressive & pass_mask).sum()),
                int(pass_mask.sum()),
                int((other_progressive & pass_mask_other).sum()),
                int(pass_mask_other.sum()),
            )
        )
        metrics.append(
            _rate_metric(
                "Deep completions/endings",
                int(pd.to_numeric(selected[end_x_col], errors="coerce").ge(80).sum()),
                len(selected),
                int(pd.to_numeric(other[end_x_col], errors="coerce").ge(80).sum()) if not other.empty else 0,
                len(other),
            )
        )

    for name, alias in [
        ("Goals", "goal"),
        ("Assists", "assist"),
        ("Key passes", "key_pass"),
    ]:
        column = find_column(frame, [alias])
        if column:
            metrics.append(
                _count_metric(
                    name,
                    int(selected[column].fillna(False).astype(bool).sum()),
                    int(other[column].fillna(False).astype(bool).sum()) if not other.empty else 0,
                )
            )

    return metrics


def _event_column(frame: pd.DataFrame) -> str | None:
    return find_column(frame, ["event_name", "eventName", "type", "event_type", "event"])


def _sub_event_column(frame: pd.DataFrame) -> str | None:
    return find_column(frame, ["sub_event_name", "subEventName", "subtype", "sub_event", "action"])


def _top_values(frame: pd.DataFrame, column: str | None, label: str) -> pd.DataFrame:
    if column is None:
        return pd.DataFrame(columns=[label, "count", "share"])
    counts = frame[column].fillna("Unknown").astype(str).value_counts().head(10)
    table = counts.rename_axis(label).reset_index(name="count")
    table["share"] = (table["count"] / len(frame)).round(3)
    return table


def _count_metric(name: str, selected_value: int, reference_value: int | None = None) -> dict[str, Any]:
    return {
        "metric": name,
        "selected": selected_value,
        "reference": reference_value,
        "selected_display": str(selected_value),
        "reference_display": "" if reference_value is None else str(reference_value),
    }


def _rate_metric(
    name: str,
    selected_numerator: int,
    selected_denominator: int,
    reference_numerator: int,
    reference_denominator: int,
) -> dict[str, Any]:
    selected_rate = _safe_rate(selected_numerator, selected_denominator)
    reference_rate = _safe_rate(reference_numerator, reference_denominator)
    return {
        "metric": name,
        "selected": selected_rate,
        "reference": reference_rate,
        "selected_display": _format_percent(selected_rate),
        "reference_display": _format_percent(reference_rate),
    }


def _mean_metric(name: str, selected_series: pd.Series, reference_series: pd.Series | None = None) -> dict[str, Any]:
    selected_mean = pd.to_numeric(selected_series, errors="coerce").mean()
    reference_mean = (
        pd.to_numeric(reference_series, errors="coerce").mean()
        if reference_series is not None and not reference_series.empty
        else None
    )
    return {
        "metric": name,
        "selected": selected_mean,
        "reference": reference_mean,
        "selected_display": _format_number(selected_mean),
        "reference_display": _format_number(reference_mean),
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1f}"


def _format_prompt_context(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    other: pd.DataFrame,
    team_name: str,
    team_column: str,
    metrics: list[dict[str, Any]],
    top_events: pd.DataFrame,
    top_sub_events: pd.DataFrame,
) -> str:
    metric_lines = [
        f"- {item['metric']}: {item['selected_display']} (other rows: {item['reference_display'] or 'n/a'})"
        for item in metrics
    ]
    event_lines = _format_table_lines(top_events, ["event", "count", "share"])
    sub_event_lines = _format_table_lines(top_sub_events, ["sub_event", "count", "share"])
    columns = ", ".join(str(column) for column in frame.columns[:30])

    return "\n".join(
        [
            "Dataset profile",
            f"- Total rows: {len(frame)}",
            f"- Selected team: {team_name}",
            f"- Team column: {team_column}",
            f"- Selected team rows: {len(selected)}",
            f"- Other rows: {len(other)}",
            f"- Available columns, truncated: {columns}",
            "",
            "Selected team metrics",
            *metric_lines,
            "",
            "Top event types",
            *event_lines,
            "",
            "Top sub-event types",
            *sub_event_lines,
        ]
    )


def _format_table_lines(table: pd.DataFrame, columns: list[str]) -> list[str]:
    if table.empty:
        return ["- n/a"]
    lines = []
    for record in table[columns].to_dict(orient="records"):
        values = ", ".join(f"{key}={value}" for key, value in record.items())
        lines.append(f"- {values}")
    return lines

