import requests
import os
import csv
import time

# Riot developer key (https://developer.riotgames.com). Dev keys expire every 24h.
# Don't commit a real key: paste it here locally or set the RIOT_API_KEY env variable.
API_KEY = os.getenv("RIOT_API_KEY", "RGAPI-a1e686c9-2a51-4a2d-8e6b-88f3477481fb")

tiers = ["GOLD", "SILVER", "PLATINUM", "BRONZE"]
divisions = ["I", "II", "III", "IV"]
queue = "RANKED_SOLO_5x5"
region = "euw1"
BASE_URL = f"https://{region}.api.riotgames.com"

MIN_MASTERY_POINTS = 30000


def riot_get(url, params=None):
    """GET a Riot API endpoint with the API key; wait and retry when rate limited (429)."""
    while True:
        r = requests.get(url, headers={"X-Riot-Token": API_KEY}, params=params)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10)))
            continue
        r.raise_for_status()
        return r.json()


"""
if the champions csv doesn't exist, download (id, name) from Data Dragon and save it.
then read the csv and return it as a dictionary {id: name}
"""
def get_champions_by_id():

    path = os.path.join("data", "champions_id.csv")

    if not os.path.exists(path):
        v = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
        data = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json").json()["data"]
        id_to_name = {int(c["key"]): c["name"] for c in data.values()}
        # save as csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            for champ_id, name in sorted(id_to_name.items()):
                writer.writerow([champ_id, name])

    # extract the champions (id,name) and return it as a dictionary
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["id"]): row["name"] for row in csv.DictReader(f)}


"""
loop over the arrays "tiers" and "division" to extract players puid from /lol/league-exp/v4/entries/{queue}/{tier}/{division}
page is a query parameter.
return a list that contains the players puuid
"""
def get_players_puuid(page = 1):
    puuids = []
    for tier in tiers:
        for division in divisions:
            url = f"{BASE_URL}/lol/league-exp/v4/entries/{queue}/{tier}/{division}"
            entries = riot_get(url, params={"page": page})
            puuids.extend(e["puuid"] for e in entries if "puuid" in e)
    return puuids


"""
get the "top" most played champion of a player using /lol/champion-mastery/v4/champion-masteries/by-puuid/{encryptedPUUID}/top
remove the champions that has less than 30000 (put that value in a variable)
return a list[int] which are the champion ids
"""
def get_top_champions(puuid, top = 5):
    url = f"{BASE_URL}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top"
    masteries = riot_get(url, params={"count": top})
    # the API already returns them sorted by mastery points (highest first)
    return [m["championId"] for m in masteries if m["championPoints"] >= MIN_MASTERY_POINTS]


"""
champions_id is a list[int].
return a dictionary{int:str} where the keys are the input keys and teh values are the values from id_name.
"""
def replace_champion_ids(champions_id):
    id_name = get_champions_by_id()
    return {champ_id: id_name[champ_id] for champ_id in champions_id if champ_id in id_name}


"""
run the whole pipeline:
1. collect players puuids from the ranked ladder
2. for each player, get his top champions (ordered by mastery points)
3. replace the champion ids by their names
4. save each player's champion "sentence" in data/player_pools.csv
"""
OUTPUT_PATH = os.path.join("data", "player_pools.csv")
MIN_CHAMPIONS = 2  # Word2Vec needs at least 2 champions in a sentence to learn from it


def main(pages = 1, top = 5):
    get_champions_by_id()  # make sure the champions csv exists before starting

    # 1. collect puuids (dict.fromkeys removes duplicates and keeps the order)
    puuids = []
    for page in range(1, pages + 1):
        puuids.extend(get_players_puuid(page))
    puuids = list(dict.fromkeys(puuids))
    print(f"{len(puuids)} players found")

    # players already saved in the csv: we don't fetch their champions again
    existing = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            existing = {row["puuid"] for row in csv.DictReader(f)}
    print(f"{len(existing)} players already in {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0
    # "a" = append to the existing csv instead of overwriting it
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # champions are separated by "|" because some names contain spaces (e.g. "Lee Sin")
        if write_header:
            writer.writerow(["puuid", "champions"])

        saved = 0
        for i, puuid in enumerate(puuids, 1):
            if puuid in existing:
                continue  # already in the csv, skip the API call
            # 2. top champions of the player
            try:
                champion_ids = get_top_champions(puuid, top)
            except requests.HTTPError as e:
                print(f"skipping player {i}: {e}")
                continue

            # 3. ids -> names (dicts keep insertion order, so the mastery order is kept)
            names = list(replace_champion_ids(champion_ids).values())

            # 4. save the player's sentence
            if len(names) >= MIN_CHAMPIONS:
                writer.writerow([puuid, "|".join(names)])
                saved += 1

            if i % 100 == 0:
                f.flush()  # write to disk regularly so progress isn't lost if it crashes
                print(f"{i}/{len(puuids)} players processed, {saved} saved")

    print(f"done: {saved} players saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main(10)