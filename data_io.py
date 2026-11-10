"""Shared file and network helpers for the two collection scripts."""

import csv
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
ABILITIES = ("P", "Q", "W", "E", "R")
RAW_FIELDS = ("champion", *ABILITIES)
CLEAN_FIELDS = (
    "champion", *(f"{slot}_{kind}" for slot in ABILITIES for kind in ("text", "tokens")),
    "kit_text", "kit_tokens",
)
PLAYER_FIELDS = ("puuid", "champions")
DDRAGON = "https://ddragon.leagueoflegends.com"


def get_json(url, *, headers=None, params=None):
    """Retry temporary failures and bound each request."""
    for attempt in range(4):
        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=20, allow_redirects=not headers,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            continue
        if headers and 300 <= response.status_code < 400:
            raise requests.HTTPError("Authenticated API request redirected.", response=response)
        if attempt < 3 and response.status_code in (429, 500, 502, 503, 504):
            retry_after = response.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            if delay > 120:
                response.raise_for_status()
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response.json()


def read_rows(path, expected_header=()):
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, strict=True)
        fields = reader.fieldnames or []
        if (not fields or len(fields) != len(set(fields))
                or (expected_header and tuple(fields) != expected_header)):
            raise ValueError(f"{Path(path).name} has a missing or invalid CSV header.")
        rows = []
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{Path(path).name}, line {reader.line_num}: wrong number of columns.")
            rows.append(row)
        return rows


def read_champion_ids(path):
    catalog = {}
    names = set()
    for row in read_rows(path, ("id", "name")):
        number = int(row["id"])
        name = row["name"]
        if (number <= 0 or number in catalog or not name.strip()
                or name != name.strip() or name.casefold() in names):
            raise ValueError("The champion catalog requires distinct positive IDs and names.")
        catalog[number] = name
        names.add(name.casefold())
    if not catalog:
        raise ValueError("The champion catalog is empty.")
    return catalog


def read_player_pools(path):
    """Check player identities before any train/test split."""
    pools = {}
    for number, row in enumerate(read_rows(path, PLAYER_FIELDS), 2):
        player = row["puuid"]
        names = row["champions"].split("|")
        if not player.strip() or player != player.strip() or player in pools:
            raise ValueError(f"Player row {number} has a blank, malformed or duplicate ID.")
        if (not 2 <= len(names) <= 5 or len({name.casefold() for name in names}) != len(names)
                or any(not name or name != name.strip() for name in names)):
            raise ValueError(f"Player row {number} must contain 2 to 5 distinct champion names.")
        pools[player] = names
    return pools


def load_abilities(path=DATA_DIR / "champions_clean.csv"):
    """Load the natural text consumed by similarity and tagging."""
    return {
        row["champion"]: {slot: row[f"{slot}_text"] for slot in ABILITIES}
        for row in read_rows(path, CLEAN_FIELDS)
    }


def resolve_champion(name, catalog):
    key = name.strip().casefold()
    for canonical in catalog:
        if canonical.casefold() == key:
            return canonical
    raise ValueError(f"Unknown champion: {name}")


def write_rows(path, fields, rows):
    """Replace a CSV only after its complete replacement has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            file.close()
            temporary.replace(path)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)
