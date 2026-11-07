import os
import re
import csv
import html
import requests
import pandas as pd
import spacy

DDRAGON = "https://ddragon.leagueoflegends.com"
CHAMPIONS_PATH = os.path.join("data", "champions_id.csv")
OUTPUT_PATH = os.path.join("data", "champion_descriptions.csv")

ABILITIES = ["P", "Q", "W", "E", "R"]

# ---------------------------------------------------------------------------
# Editable preprocessing dictionaries
# ---------------------------------------------------------------------------

# multi-word game terms -> one token (regex pattern -> replacement).
# applied on the lowercased text BEFORE tokenizing, so spaCy keeps them together.
PHRASES = {
    r"\bknock(?:s|ed|ing)?\s+(?:\w+\s+)?up\b": "knockup",          # "knocks up", "knocking enemies up"
    r"\bknock(?:s|ed|ing)?\s+(?:\w+\s+)?(?:back|aside|away)\b": "knockback",
    r"\b(?:up\s+)?(?:in|into)\s+the\s+air\b": "airborne",          # "pop their target up in the air"
    r"\bcrowd\s+control\b": "crowdcontrol",
    r"\bmove(?:ment)?\s+speed\b": "movespeed",
    r"\battack\s+speed\b": "attackspeed",
    r"\battack\s+damage\b": "attackdamage",
    r"\bability\s+power\b": "abilitypower",
    r"\bmagic\s+damage\b": "magicdamage",
    r"\bphysical\s+damage\b": "physicaldamage",
    r"\btrue\s+damage\b": "truedamage",
    r"\bbasic\s+attacks?\b": "basicattack",
    r"\bon-hit\b": "onhit",
    r"\blife\s*steal\b": "lifesteal",
    r"\bcritical\s+strikes?\b": "crit",
    r"\bdamage\s+over\s+time\b": "dot",
}

# words that are very frequent in ability descriptions but say nothing about the kit
# (removed on top of spaCy's standard English stopwords)
CUSTOM_STOPWORDS = {
    "enemy", "ability", "deal", "champion", "second", "nearby", "target", "unit",
    "cast", "time", "additionally", "grant", "gain", "use", "hit", "effect",
    "duration", "amount", "level", "short", "brief", "briefly",
}

# lemma -> shared concept, so different words for the same mechanic become one token
SYNONYMS = {
    # mobility
    "blink": "dash", "leap": "dash", "lunge": "dash", "jump": "dash", "rush": "dash",
    # displacement
    "airborne": "knockup",
    "grab": "pull", "drag": "pull", "hook": "pull",
    # crowd control
    "snare": "root", "immobilize": "root", "bind": "root",
    # protection / sustain
    "barrier": "shield",
    "healing": "heal", "restore": "heal", "regenerate": "heal",
    # stealth
    "invisible": "stealth", "invisibility": "stealth", "camouflage": "stealth",
}

# parts of multi-word champion names that are normal English words:
# we don't replace them alone (e.g. "Twisted" from "Twisted Fate")
NAME_PARTS_EXCLUDE = {"twisted", "master", "miss", "dr", "iv"}

# spaCy model loaded once for the whole module (parser and NER aren't needed for lemmas)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOPWORDS = nlp.Defaults.stop_words | CUSTOM_STOPWORDS


# ---------------------------------------------------------------------------
# Light cleaning (-> X_text)
# ---------------------------------------------------------------------------

"""
step 1: remove HTML / markup tags (<br>, <magicDamage>, <font ...>) and template
placeholders ({{ e1 }}, @Effect1Amount@). Tags become a space so words don't stick together.
"""
def remove_markup(text):
    text = html.unescape(text)                  # &nbsp; &amp; ... -> normal characters
    text = re.sub(r"<[^>]+>", " ", text)        # HTML / markup tags
    text = re.sub(r"\{\{.*?\}\}", " ", text)    # {{ e1 }}
    text = re.sub(r"@[\w.*:]+@", " ", text)     # @Effect1Amount@
    return text


"""
step 2: replace the champion's name with "the champion" (case-insensitive).
the full name is replaced first, then its parts ("Nunu & Willump" -> "Nunu", "Willump";
"Lee Sin" -> "Lee", "Sin"). Apostrophes stay inside the name ("Kai'Sa", "Kai'Sa's").
"""
def name_variants(name):
    parts = [p.strip(".") for p in re.split(r"[\s&]+", name) if p.strip(".")]
    parts = [p for p in parts if p.lower() not in NAME_PARTS_EXCLUDE]
    # longest first, so "Lee Sin" is replaced before "Lee"
    return sorted({name, *parts}, key=len, reverse=True)


def replace_champion_name(text, name):
    def repl(m):
        # capitalize when the name starts a sentence
        before = m.string[:m.start()].rstrip()
        return "The champion" if before == "" or before[-1] in ".!?:" else "the champion"

    for variant in name_variants(name):
        # apostrophes can be written ' or ’ in the data
        pattern = re.escape(variant).replace("'", "['’]")
        text = re.sub(rf"(?<!\w){pattern}(?!\w)", repl, text, flags=re.IGNORECASE)
    return text


"""
step 3: remove numbers and percentages ("50%", "1.5", "4").
"""
def remove_numbers(text):
    return re.sub(r"\d+(?:[.,]\d+)?\s*%?", " ", text)


"""
step 4: collapse repeated whitespace, remove spaces before punctuation, strip.
"""
def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    return text.strip()


"""
light cleaning: natural sentences for Sentence-BERT / zero-shot classification
(no lowercasing, no stopword removal, no lemmatization)
"""
def light_clean(text, champion):
    if not isinstance(text, str) or not text.strip():   # NaN / empty cell
        return ""
    text = remove_markup(text)
    text = replace_champion_name(text, champion)
    text = remove_numbers(text)
    return normalize_whitespace(text)


# ---------------------------------------------------------------------------
# Full preprocessing (-> X_tokens), starting from the light-cleaned text
# ---------------------------------------------------------------------------

"""
step 6: replace multi-word game terms with one token ("knocked up" -> "knockup")
"""
def replace_phrases(text):
    for pattern, token in PHRASES.items():
        text = re.sub(pattern, token, text)
    return text


"""
steps 7 + 8: tokenize and lemmatize with spaCy, keep only alphabetic tokens,
remove standard + custom stopwords
"""
def lemmatize(text):
    lemmas = []
    for tok in nlp(text):
        lemma = tok.lemma_.lower()
        if tok.is_alpha and lemma not in STOPWORDS and tok.lower_ not in STOPWORDS:
            lemmas.append(lemma)
    return lemmas


"""
step 9: map each lemma to its shared concept with the synonym dictionary
"""
def apply_synonyms(lemmas):
    return [SYNONYMS.get(lemma, lemma) for lemma in lemmas]


"""
full preprocessing: space-separated tokens for TF-IDF / keyword tagging
"""
def full_preprocess(clean_text):
    if not clean_text:
        return ""
    text = clean_text.lower()      # step 5
    text = replace_phrases(text)   # step 6
    lemmas = lemmatize(text)       # steps 7 + 8
    return " ".join(apply_synonyms(lemmas))   # step 9


"""
read the raw descriptions csv (champion,P,Q,W,E,R), preprocess every ability and
save the result to champions_clean.csv in the same folder.
output columns: champion, X_text, X_tokens for X in P,Q,W,E,R, kit_text, kit_tokens.
return the resulting DataFrame
"""
def clean(path):
    raw = pd.read_csv(path)
    name_col = raw.columns[0]   # "champion" (first column)

    rows = []
    for _, row in raw.iterrows():
        champion = row[name_col]
        out = {name_col: champion}
        for x in ABILITIES:
            out[f"{x}_text"] = light_clean(row[x], champion)
            out[f"{x}_tokens"] = full_preprocess(out[f"{x}_text"])
        # whole kit = the 5 abilities joined (empty abilities skipped)
        out["kit_text"] = " ".join(out[f"{x}_text"] for x in ABILITIES if out[f"{x}_text"])
        out["kit_tokens"] = " ".join(out[f"{x}_tokens"] for x in ABILITIES if out[f"{x}_tokens"])
        rows.append(out)

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(path), "champions_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"cleaned {len(df)} champions, saved to {out_path}")
    return df

"""
read the champions display names from data/champions_id.csv (created by champion_collection.py)
"""
def get_champion_names():
    with open(CHAMPIONS_PATH, newline="", encoding="utf-8") as f:
        return [row["name"] for row in csv.DictReader(f)]


"""
Data Dragon uses internal ids in its urls, which can differ from the display names
(e.g. "Wukong" -> "MonkeyKing", "Lee Sin" -> "LeeSin").
return a dictionary {display name: internal id}
"""
def get_name_to_ddragon_id(patch):
    data = requests.get(f"{DDRAGON}/cdn/{patch}/data/en_US/champion.json").json()["data"]
    return {c["name"]: c["id"] for c in data.values()}


"""
extract the passive and the 4 spells descriptions from a champion json.
return a list [P, Q, W, E, R]
"""
def extract_abilities(champ_json, ddragon_id):
    champ = champ_json["data"][ddragon_id]
    passive = champ["passive"]["description"]
    # "spells" is always ordered Q, W, E, R
    spells = [spell["description"] for spell in champ["spells"]]
    return [passive] + spells


def main():
    # get all champions names from data/champions_id.csv
    champions = get_champion_names()

    latest_patch = requests.get(f"{DDRAGON}/api/versions.json").json()[0]
    name_to_id = get_name_to_ddragon_id(latest_patch)
    print(f"patch {latest_patch}: {len(champions)} champions")

    # create a new csv with the columns : champion,P,Q,W,E,R
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["champion", "P", "Q", "W", "E", "R"])

        for i, champ in enumerate(champions, 1):
            ddragon_id = name_to_id.get(champ)
            if ddragon_id is None:
                print(f"skipping {champ}: not found in Data Dragon")
                continue

            # download champion json
            url = f"{DDRAGON}/cdn/{latest_patch}/data/en_US/champion/{ddragon_id}.json"
            try:
                r = requests.get(url)
                r.raise_for_status()
                champ_json = r.json()
            except requests.RequestException as e:
                print(f"skipping {champ}: {e}")
                continue

            # extract the abilities and passive description
            abilities = extract_abilities(champ_json, ddragon_id)

            # append the csv with the current data
            # the display name is kept so the ids match player_pools.csv
            writer.writerow([champ] + abilities)

            if i % 20 == 0:
                print(f"{i}/{len(champions)} champions done")

    print(f"done: saved to {OUTPUT_PATH}")

    # preprocess the raw descriptions -> data/champions_clean.csv
    clean(OUTPUT_PATH)


if __name__ == "__main__":
    main()
