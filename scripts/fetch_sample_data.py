from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_URL = "https://raw.githubusercontent.com/koenvo/wyscout-soccer-match-event-dataset/main/processed/files/2499754.json"
RAW_OUTPUT = ROOT / "data" / "raw" / "wyscout_2499754_liverpool_manchester_city.json"
CSV_OUTPUT = ROOT / "data" / "sample" / "wyscout_2499754_events.csv"
JSON_OUTPUT = ROOT / "data" / "sample" / "wyscout_2499754_events.json"
NOTES_OUTPUT = ROOT / "data" / "sample" / "wyscout_2499754_notes.md"


def main() -> None:
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {RAW_URL}")
    with urllib.request.urlopen(RAW_URL, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))

    RAW_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = wyscout_rows(payload)
    write_csv(CSV_OUTPUT, rows)
    JSON_OUTPUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    NOTES_OUTPUT.write_text(notes_text(), encoding="utf-8")

    print(f"Wrote {RAW_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {CSV_OUTPUT.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"Wrote {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {NOTES_OUTPUT.relative_to(ROOT)}")


def wyscout_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    team_lookup = extract_team_lookup(payload)
    player_lookup = extract_player_lookup(payload)
    rows = []
    for event in payload["events"]:
        positions = event.get("positions", [])
        start = positions[0] if len(positions) > 0 else {}
        end = positions[1] if len(positions) > 1 else {}
        tag_ids = [
            int(tag["id"])
            for tag in event.get("tags", [])
            if isinstance(tag, dict) and str(tag.get("id", "")).isdigit()
        ]
        team_id = event.get("teamId")
        player_id = event.get("playerId")
        rows.append(
            {
                "match_id": event.get("matchId"),
                "event_id": event.get("id"),
                "period": event.get("matchPeriod"),
                "second": event.get("eventSec"),
                "team_id": team_id,
                "team_name": team_lookup.get(str(team_id), str(team_id)),
                "player_id": player_id,
                "player_name": player_lookup.get(str(player_id), str(player_id)),
                "event_name": event.get("eventName"),
                "sub_event_name": event.get("subEventName"),
                "start_x": start.get("x"),
                "start_y": start.get("y"),
                "end_x": end.get("x"),
                "end_y": end.get("y"),
                "tag_ids": "|".join(str(tag_id) for tag_id in tag_ids),
                "accurate": 1801 in tag_ids,
                "not_accurate": 1802 in tag_ids,
                "goal": 101 in tag_ids,
                "assist": 301 in tag_ids,
                "key_pass": 302 in tag_ids,
                "source_url": RAW_URL,
            }
        )
    return rows


def extract_team_lookup(payload: dict[str, Any]) -> dict[str, str]:
    lookup = {}
    for key, value in payload.get("teams", {}).items():
        team = value.get("team", value)
        team_id = value.get("teamId") or team.get("wyId") or key
        lookup[str(team_id)] = team.get("name") or team.get("officialName") or str(team_id)
    return lookup


def extract_player_lookup(payload: dict[str, Any]) -> dict[str, str]:
    lookup = {}
    for player_list in payload.get("players", {}).values():
        for entry in player_list:
            player = entry.get("player", {})
            player_id = entry.get("playerId") or player.get("wyId")
            if player_id is None:
                continue
            lookup[str(player_id)] = player.get("shortName") or str(player_id)
    return lookup


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def notes_text() -> str:
    return """# Sample scouting notes

Source match: Manchester City 5-0 Liverpool, Premier League 2017/18, match id 2499754.

The structured sample is derived from the public Wyscout event JSON in the koenvo/wyscout-soccer-match-event-dataset repository. It is useful for demonstrating how a language model can combine aggregate event evidence with short analyst notes. It is not enough to make season-level claims.
"""


if __name__ == "__main__":
    main()

