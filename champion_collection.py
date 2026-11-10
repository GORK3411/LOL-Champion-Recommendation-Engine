"""Collect EUW ranked players' most played champions by mastery."""

import argparse
import csv
import os
from pathlib import Path

import requests

from data_io import DATA_DIR, DDRAGON, PLAYER_FIELDS, get_json, read_champion_ids, read_player_pools, write_rows

TIERS = ("GOLD", "SILVER", "PLATINUM", "BRONZE")
DIVISIONS = ("I", "II", "III", "IV")
QUEUE = "RANKED_SOLO_5x5"
BASE_URL = "https://euw1.api.riotgames.com"
MIN_MASTERY_POINTS = 30_000
MIN_CHAMPIONS = 2
CHAMPIONS_PATH = DATA_DIR / "champions_id.csv"
OUTPUT_PATH = DATA_DIR / "player_pools.csv"


def riot_get(url, params=None):
    key = os.getenv("RIOT_API_KEY", "").strip()
    if not key:
        raise ValueError("Set RIOT_API_KEY before collecting player data.")
    return get_json(url, headers={"X-Riot-Token": key}, params=params)


def get_champions_by_id():
    if not Path(CHAMPIONS_PATH).exists():
        patch = get_json(f"{DDRAGON}/api/versions.json")[0]
        champions = get_json(f"{DDRAGON}/cdn/{patch}/data/en_US/champion.json")["data"]
        rows = [
            {"id": int(champion["key"]), "name": champion["name"]}
            for champion in champions.values()
        ]
        write_rows(CHAMPIONS_PATH, ("id", "name"), sorted(rows, key=lambda row: row["id"]))
    return read_champion_ids(CHAMPIONS_PATH)


def get_players_puuid(page=1):
    puuids = []
    for tier in TIERS:
        for division in DIVISIONS:
            url = f"{BASE_URL}/lol/league-exp/v4/entries/{QUEUE}/{tier}/{division}"
            entries = riot_get(url, params={"page": page})
            puuids.extend(entry["puuid"] for entry in entries if entry.get("puuid"))
    return puuids


def get_top_champions(puuid, top=5):
    url = f"{BASE_URL}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top"
    mastery = riot_get(url, params={"count": top})
    return [
        row["championId"]
        for row in sorted(mastery, key=lambda row: row["championPoints"], reverse=True)[:top]
        if row["championPoints"] >= MIN_MASTERY_POINTS
    ]


def replace_champion_ids(champion_ids, names_by_id=None):
    if names_by_id is None:
        names_by_id = get_champions_by_id()
    return {number: names_by_id[number] for number in champion_ids if number in names_by_id}


def main(pages=1, top=5):
    if pages < 1 or not 2 <= top <= 5:
        raise ValueError("Use at least one page and a top count from 2 to 5.")
    if not os.getenv("RIOT_API_KEY", "").strip():
        raise ValueError("Set RIOT_API_KEY before collecting player data.")

    names_by_id = get_champions_by_id()
    players = dict.fromkeys(
        puuid for page in range(1, pages + 1) for puuid in get_players_puuid(page)
    )
    path = Path(OUTPUT_PATH)
    has_file = path.exists() and path.stat().st_size > 0
    existing = read_player_pools(path) if has_file else {}
    pending = [puuid for puuid in players if puuid not in existing]
    print(f"{len(players)} players found, {len(pending)} to fetch")
    if not pending:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    # An edited CSV can end with a valid record that has no line terminator.
    needs_newline = False
    if has_file:
        with path.open("rb") as file:
            file.seek(-1, os.SEEK_END)
            needs_newline = file.read(1) not in (b"\r", b"\n")
    saved = 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if needs_newline:
            file.write("\r\n")
        if not has_file:
            writer.writerow(PLAYER_FIELDS)
        for index, puuid in enumerate(pending, 1):
            try:
                ids = get_top_champions(puuid, top)
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 404:
                    raise
                ids = []  # The ladder can include a player whose mastery record is gone.
            names = list(replace_champion_ids(ids, names_by_id).values())
            if len(names) >= MIN_CHAMPIONS:
                writer.writerow((puuid, "|".join(names)))
                saved += 1
            if index % 100 == 0:
                file.flush()
                print(f"{index}/{len(pending)} checked, {saved} saved")
    print(f"Saved {saved} new player pools to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    try:
        main(args.pages, args.top)
    except (ValueError, OSError, csv.Error, requests.RequestException) as error:
        parser.exit(1, f"Collection stopped: {error}\n")
