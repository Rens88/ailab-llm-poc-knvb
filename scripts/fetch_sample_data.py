from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_BASE_URL = "https://raw.githubusercontent.com/koenvo/wyscout-soccer-match-event-dataset/main/processed/files"
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"
MATCH_SAMPLE_DIR = SAMPLE_DIR / "matches"
NOTES_OUTPUT = SAMPLE_DIR / "wyscout_2499754_notes.md"
COMBINED_CSV_OUTPUT = SAMPLE_DIR / "wyscout_multi_match_events.csv"
COMBINED_JSON_OUTPUT = SAMPLE_DIR / "wyscout_multi_match_events.json"
SINGLE_CSV_OUTPUT = SAMPLE_DIR / "wyscout_2499754_events.csv"
SINGLE_JSON_OUTPUT = SAMPLE_DIR / "wyscout_2499754_events.json"


@dataclass(frozen=True)
class MatchMeta:
    match_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    date: str
    competition: str = "English Premier League"
    season: str = "2017/18"

    @property
    def label(self) -> str:
        return f"{self.home_team} - {self.away_team}, {self.home_score} - {self.away_score}"

    @property
    def slug(self) -> str:
        teams = f"{self.home_team}_{self.away_team}".lower()
        return "".join(character if character.isalnum() else "_" for character in teams).strip("_")


SAMPLE_MATCHES = [
    MatchMeta(2499720, "Brighton & Hove Albion", "Manchester City", 0, 2, "2017-08-12"),
    MatchMeta(2499734, "Manchester City", "Everton", 1, 1, "2017-08-21"),
    MatchMeta(2499739, "AFC Bournemouth", "Manchester City", 1, 2, "2017-08-26"),
    MatchMeta(2499754, "Manchester City", "Liverpool", 5, 0, "2017-09-09"),
    MatchMeta(2499767, "Watford", "Manchester City", 0, 6, "2017-09-16"),
    MatchMeta(2499774, "Manchester City", "Crystal Palace", 5, 0, "2017-09-23"),
    MatchMeta(2499781, "Chelsea", "Manchester City", 0, 1, "2017-09-30"),
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    MATCH_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []

    for meta in SAMPLE_MATCHES:
        url = f"{RAW_BASE_URL}/{meta.match_id}.json"
        csv_output = MATCH_SAMPLE_DIR / f"wyscout_{meta.match_id}_{meta.slug}.csv"

        print(f"Fetching {url}")
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if meta.match_id == 2499754:
            raw_output = RAW_DIR / "wyscout_2499754_liverpool_manchester_city.json"
            raw_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rows = wyscout_rows(payload, meta, url)
        write_csv(csv_output, rows)
        combined_rows.extend(rows)
        if meta.match_id == 2499754:
            single_rows = rows
        print(f"Wrote {csv_output.relative_to(ROOT)} ({len(rows)} rows)")

    write_csv(COMBINED_CSV_OUTPUT, combined_rows)
    COMBINED_JSON_OUTPUT.write_text(json.dumps(combined_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(SINGLE_CSV_OUTPUT, single_rows)
    SINGLE_JSON_OUTPUT.write_text(json.dumps(single_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    NOTES_OUTPUT.write_text(notes_text(), encoding="utf-8")

    print(f"Wrote {COMBINED_CSV_OUTPUT.relative_to(ROOT)} ({len(combined_rows)} rows)")
    print(f"Wrote {COMBINED_JSON_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {SINGLE_CSV_OUTPUT.relative_to(ROOT)} ({len(single_rows)} rows)")
    print(f"Wrote {SINGLE_JSON_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {NOTES_OUTPUT.relative_to(ROOT)}")


def wyscout_rows(payload: dict[str, Any], meta: MatchMeta, source_url: str) -> list[dict[str, Any]]:
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
        event_name = event.get("eventName")
        sub_event_name = event.get("subEventName")
        is_shot = str(event_name).lower() == "shot" or str(sub_event_name).lower() == "shot"
        rows.append(
            {
                "match_id": event.get("matchId"),
                "match_label": meta.label,
                "match_date": meta.date,
                "competition": meta.competition,
                "season": meta.season,
                "home_team": meta.home_team,
                "away_team": meta.away_team,
                "home_score": meta.home_score,
                "away_score": meta.away_score,
                "event_id": event.get("id"),
                "period": event.get("matchPeriod"),
                "second": event.get("eventSec"),
                "team_id": team_id,
                "team_name": team_lookup.get(str(team_id), str(team_id)),
                "player_id": player_id,
                "player_name": player_lookup.get(str(player_id), str(player_id)),
                "event_name": event_name,
                "sub_event_name": sub_event_name,
                "start_x": start.get("x"),
                "start_y": start.get("y"),
                "end_x": end.get("x"),
                "end_y": end.get("y"),
                "tag_ids": "|".join(str(tag_id) for tag_id in tag_ids),
                "accurate": 1801 in tag_ids,
                "not_accurate": 1802 in tag_ids,
                "goal": is_shot and 101 in tag_ids,
                "shot_on_target": is_shot and (1801 in tag_ids or 101 in tag_ids),
                "assist": 301 in tag_ids,
                "key_pass": 302 in tag_ids,
                "source_url": source_url,
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
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def notes_text() -> str:
    return """# Sample scouting notes

Source samples: seven Manchester City matches from the English Premier League 2017/18 segment of the public Wyscout event dataset.

The structured sample is derived from the koenvo/wyscout-soccer-match-event-dataset repository. It is useful for demonstrating how a language model can combine aggregate event evidence with short analyst notes. The multi-match sample is still a small early-season slice, not a full-season scouting report.
"""


if __name__ == "__main__":
    main()
