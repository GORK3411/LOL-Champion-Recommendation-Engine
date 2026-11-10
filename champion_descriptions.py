"""Download champion ability text and prepare two forms for NLP."""

import argparse
import csv
from functools import cache
import html
from pathlib import Path
import re
from shutil import copyfile
from tempfile import TemporaryDirectory

import pandas as pd
import requests

from data_io import (
    ABILITIES, CLEAN_FIELDS, DATA_DIR, DDRAGON, RAW_FIELDS,
    get_json, read_champion_ids, read_rows, write_rows,
)

CHAMPIONS_PATH = DATA_DIR / "champions_id.csv"
OUTPUT_PATH = DATA_DIR / "champion_descriptions.csv"

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

@cache
def get_nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


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
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)    # {{ e1 }}
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
    nlp = get_nlp()
    stopwords = nlp.Defaults.stop_words | CUSTOM_STOPWORDS
    lemmas = []
    for tok in nlp(text):
        lemma = tok.lemma_.lower()
        if tok.is_alpha and lemma not in stopwords and tok.lower_ not in stopwords:
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


def preprocess_rows(raw_rows):
    """Produce natural text, normalized tokens, and joined kits."""
    rows = []
    seen = set()
    for row in raw_rows:
        champion = row["champion"]
        if not champion.strip() or champion != champion.strip() or champion in seen:
            raise ValueError("Description rows require distinct, nonempty champion names.")
        seen.add(champion)
        out = {"champion": champion}
        for slot in ABILITIES:
            out[f"{slot}_text"] = light_clean(row[slot], champion)
            if not out[f"{slot}_text"]:
                raise ValueError(f"{champion} {slot} has no text after cleaning.")
            out[f"{slot}_tokens"] = full_preprocess(out[f"{slot}_text"])
        for kind in ("text", "tokens"):
            out[f"kit_{kind}"] = " ".join(
                out[f"{slot}_{kind}"] for slot in ABILITIES if out[f"{slot}_{kind}"]
            )
        rows.append(out)
    if not rows:
        raise ValueError("No champion descriptions to clean.")
    return rows


def clean(path):
    """Regenerate champions_clean.csv next to the raw CSV."""
    rows = preprocess_rows(read_rows(path, RAW_FIELDS))
    output = Path(path).with_name("champions_clean.csv")
    write_rows(output, CLEAN_FIELDS, rows)
    print(f"Cleaned {len(rows)} champions into {output}")
    return pd.DataFrame(rows)


def get_champion_names():
    return list(read_champion_ids(CHAMPIONS_PATH).values())


def get_name_to_ddragon_id(patch):
    catalog = get_json(f"{DDRAGON}/cdn/{patch}/data/en_US/champion.json")["data"]
    return {champion["name"]: champion["id"] for champion in catalog.values()}


def extract_abilities(champ_json, ddragon_id):
    champion = champ_json["data"][ddragon_id]
    passive = champion["passive"]["description"]
    return [passive] + [spell["description"] for spell in champion["spells"]]


def publish_descriptions(raw, cleaned):
    """Stage both CSVs and restore the previous raw file if publishing clean fails."""
    raw_path = Path(OUTPUT_PATH)
    clean_path = raw_path.with_name("champions_clean.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=raw_path.parent) as folder:
        staged_raw = Path(folder) / raw_path.name
        staged_clean = Path(folder) / clean_path.name
        backup = Path(folder) / "previous-raw.csv"
        write_rows(staged_raw, RAW_FIELDS, raw)
        write_rows(staged_clean, CLEAN_FIELDS, cleaned)
        if raw_path.exists():
            copyfile(raw_path, backup)
        staged_raw.replace(raw_path)
        try:
            staged_clean.replace(clean_path)
        except OSError:
            if backup.exists():
                backup.replace(raw_path)
            else:
                raw_path.unlink()
            raise


def main(patch=None):
    champions = get_champion_names()
    patch = patch or get_json(f"{DDRAGON}/api/versions.json")[0]
    name_to_id = get_name_to_ddragon_id(patch)
    missing = set(champions) - name_to_id.keys()
    if missing:
        raise ValueError(f"Patch {patch} is missing: {', '.join(sorted(missing))}")

    print(f"Downloading {len(champions)} champions for patch {patch}")
    rows = []
    for champion in champions:
        champion_id = name_to_id[champion]
        url = f"{DDRAGON}/cdn/{patch}/data/en_US/champion/{champion_id}.json"
        abilities = extract_abilities(get_json(url), champion_id)
        if len(abilities) != len(ABILITIES) or not all(
            isinstance(text, str) and text.strip() for text in abilities
        ):
            raise ValueError(f"{champion} does not have five populated ability descriptions.")
        rows.append({"champion": champion, **dict(zip(ABILITIES, abilities))})

    # Finish all downloads and NLP before touching either existing file.
    cleaned = preprocess_rows(rows)
    publish_descriptions(rows, cleaned)
    print(f"Saved raw and cleaned descriptions for patch {patch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", help="Data Dragon version to download")
    parser.add_argument("--clean-only", action="store_true", help="Reuse the saved raw CSV")
    args = parser.parse_args()
    try:
        if args.clean_only:
            clean(OUTPUT_PATH)
        else:
            main(args.patch)
    except (ValueError, OSError, KeyError, csv.Error, requests.RequestException) as error:
        parser.exit(1, f"Description refresh stopped: {error}\n")
