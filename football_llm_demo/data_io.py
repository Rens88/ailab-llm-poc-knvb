from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


@dataclass(frozen=True)
class LoadedTable:
    name: str
    dataframe: pd.DataFrame
    source_kind: str
    metadata: dict[str, Any]


def load_table_from_path(path: Path) -> LoadedTable:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return LoadedTable(path.name, _stringify_nested_values(frame), "csv", {})

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _load_json_payload(payload, path.name)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_table_from_upload(uploaded_file: BinaryIO, filename: str) -> LoadedTable:
    suffix = Path(filename).suffix.lower()
    raw = uploaded_file.read()

    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(raw))
        return LoadedTable(filename, _stringify_nested_values(frame), "csv", {})

    if suffix == ".json":
        payload = json.loads(raw.decode("utf-8"))
        return _load_json_payload(payload, filename)

    raise ValueError(f"Unsupported file type: {suffix}")


def infer_team_column(frame: pd.DataFrame) -> str | None:
    aliases = [
        "team_name",
        "team",
        "club",
        "squad",
        "official_team_name",
        "team_id",
        "teamid",
        "wy_team_id",
    ]
    return find_column(frame, aliases)


def team_values(frame: pd.DataFrame, team_column: str) -> list[str]:
    values = frame[team_column].dropna().astype(str).unique().tolist()
    return sorted(values, key=lambda value: (len(value), value.lower()))


def find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_aliases = {_normalize_name(alias) for alias in aliases}
    for column in frame.columns:
        if _normalize_name(str(column)) in normalized_aliases:
            return str(column)
    return None


def _load_json_payload(payload: Any, filename: str) -> LoadedTable:
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        frame, metadata = _wyscout_payload_to_frame(payload)
        return LoadedTable(filename, frame, "wyscout_json", metadata)

    if isinstance(payload, list):
        frame = pd.json_normalize(payload)
        return LoadedTable(filename, _stringify_nested_values(frame), "json_records", {})

    if isinstance(payload, dict):
        key, records = _pick_record_list(payload)
        if records is not None:
            frame = pd.json_normalize(records)
            metadata = {"json_record_key": key}
            return LoadedTable(filename, _stringify_nested_values(frame), "json_records", metadata)
        frame = pd.json_normalize(payload)
        return LoadedTable(filename, _stringify_nested_values(frame), "json_object", {})

    raise ValueError("JSON root must be an object or a list.")


def _pick_record_list(payload: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None]:
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for key, value in payload.items():
        if not isinstance(value, list) or not value:
            continue
        sample = value[: min(25, len(value))]
        if all(isinstance(item, dict) for item in sample):
            candidates.append((key, value))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0]


def _wyscout_payload_to_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    teams = _extract_team_lookup(payload)
    players = _extract_player_lookup(payload)
    rows: list[dict[str, Any]] = []

    for event in payload.get("events", []):
        tag_ids = _extract_tag_ids(event.get("tags"))
        team_id = event.get("teamId")
        player_id = event.get("playerId")
        event_name = event.get("eventName")
        sub_event_name = event.get("subEventName")
        is_shot = str(event_name).lower() == "shot" or str(sub_event_name).lower() == "shot"
        start_position = _position(event.get("positions"), 0)
        end_position = _position(event.get("positions"), 1)

        rows.append(
            {
                "match_id": event.get("matchId"),
                "event_id": event.get("id"),
                "period": event.get("matchPeriod"),
                "second": event.get("eventSec"),
                "team_id": team_id,
                "team_name": teams.get(str(team_id), str(team_id)),
                "player_id": player_id,
                "player_name": players.get(str(player_id), str(player_id)),
                "event_name": event_name,
                "sub_event_name": sub_event_name,
                "event_type_id": event.get("eventId"),
                "sub_event_type_id": event.get("subEventId"),
                "start_x": start_position.get("x"),
                "start_y": start_position.get("y"),
                "end_x": end_position.get("x"),
                "end_y": end_position.get("y"),
                "tag_ids": "|".join(str(tag_id) for tag_id in tag_ids),
                "accurate": 1801 in tag_ids,
                "not_accurate": 1802 in tag_ids,
                "goal": is_shot and 101 in tag_ids,
                "assist": 301 in tag_ids,
                "key_pass": 302 in tag_ids,
            }
        )

    frame = pd.DataFrame(rows)
    metadata = {
        "teams": teams,
        "players": players,
        "match": payload.get("match", {}),
    }
    return frame, metadata


def _extract_team_lookup(payload: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    teams = payload.get("teams", {})

    if isinstance(teams, dict):
        iterable = teams.items()
    elif isinstance(teams, list):
        iterable = [(None, item) for item in teams]
    else:
        iterable = []

    for key, value in iterable:
        team_node = value.get("team", value) if isinstance(value, dict) else {}
        if not isinstance(team_node, dict):
            continue
        team_id = (
            value.get("teamId")
            if isinstance(value, dict)
            else None
        ) or team_node.get("wyId") or team_node.get("teamId") or key
        if team_id is None:
            continue
        name = team_node.get("name") or team_node.get("officialName") or str(team_id)
        lookup[str(team_id)] = str(name)

    return lookup


def _extract_player_lookup(payload: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    players = payload.get("players", {})

    entries: list[Any] = []
    if isinstance(players, dict):
        for value in players.values():
            if isinstance(value, list):
                entries.extend(value)
            elif isinstance(value, dict):
                entries.append(value)
    elif isinstance(players, list):
        entries = players

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        player_node = entry.get("player", entry)
        if not isinstance(player_node, dict):
            continue
        player_id = entry.get("playerId") or player_node.get("wyId")
        if player_id is None:
            continue
        name = (
            player_node.get("shortName")
            or " ".join(
                part
                for part in [player_node.get("firstName"), player_node.get("lastName")]
                if part
            )
            or str(player_id)
        )
        lookup[str(player_id)] = str(name)

    return lookup


def _position(positions: Any, index: int) -> dict[str, Any]:
    if isinstance(positions, list) and len(positions) > index and isinstance(positions[index], dict):
        return positions[index]
    return {}


def _extract_tag_ids(tags: Any) -> list[int]:
    tag_ids: list[int] = []
    if not isinstance(tags, list):
        return tag_ids
    for tag in tags:
        if not isinstance(tag, dict) or "id" not in tag:
            continue
        try:
            tag_ids.append(int(tag["id"]))
        except (TypeError, ValueError):
            continue
    return tag_ids


def _stringify_nested_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if cleaned[column].map(lambda value: isinstance(value, (dict, list))).any():
            cleaned[column] = cleaned[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
    return cleaned


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())
