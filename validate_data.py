"""Check that the frozen CSV snapshot is ready for all three NLP modules."""

from collections import Counter
import csv

from champion_descriptions import light_clean
from data_io import ABILITIES, CLEAN_FIELDS, DATA_DIR, RAW_FIELDS, read_champion_ids, read_player_pools, read_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_index(rows, key, label):
    index = {row[key]: row for row in rows}
    require(index and all(name.strip() and name == name.strip() for name in index)
            and len(index) == len(rows),
            f"{label} must have one row per nonempty {key}.")
    return index


def validate(directory=DATA_DIR):
    catalog = set(read_champion_ids(directory / "champions_id.csv").values())
    raw = unique_index(read_rows(directory / "champion_descriptions.csv", RAW_FIELDS), "champion", "Raw text")
    clean = unique_index(read_rows(directory / "champions_clean.csv", CLEAN_FIELDS), "champion", "Clean text")
    require(catalog == raw.keys() == clean.keys(),
            "The three champion CSVs must have the same champion names.")
    for name, row in clean.items():
        for slot in ABILITIES:
            require(row[f"{slot}_text"] and row[f"{slot}_tokens"],
                    f"{name} {slot} is missing clean text or tokens.")
            require(row[f"{slot}_text"] == light_clean(raw[name][slot], name),
                    f"{name} {slot} has mismatched raw and clean descriptions.")
        for kind in ("text", "tokens"):
            expected = " ".join(row[f"{slot}_{kind}"] for slot in ABILITIES)
            require(row[f"kit_{kind}"] == expected, f"{name} kit_{kind} does not match its slots.")

    players = read_player_pools(directory / "player_pools.csv")
    require(players, "Player pools must contain at least one row.")
    counts = Counter()
    for number, names in enumerate(players.values(), 2):
        require(set(names) <= catalog, f"Player row {number} has an unknown champion.")
        counts.update(names)
    return len(players), len(catalog), counts


if __name__ == "__main__":
    try:
        players, champions, counts = validate()
    except (ValueError, OSError, KeyError, csv.Error) as error:
        raise SystemExit(f"Data validation failed: {error}") from error
    print(f"Valid snapshot: {players:,} player pools, {champions} kits, "
          f"{champions * len(ABILITIES)} ability descriptions.")
    print(f"{len(counts)} champions appear in pools, {sum(counts.values()):,} pool entries.")
