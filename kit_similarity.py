"""Compare champion kits and match their abilities with TF-IDF and cosine."""

import argparse
from functools import cache

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer

from champion_descriptions import full_preprocess, remove_markup, remove_numbers
from data_io import ABILITIES, CLEAN_FIELDS, DATA_DIR, load_abilities, read_rows, resolve_champion

MATCH_THRESHOLD = 0.20
CLEAN_PATH = DATA_DIR / "champions_clean.csv"


@cache
def fitted_tfidf():
    rows = read_rows(CLEAN_PATH, CLEAN_FIELDS)
    tokens_by_text = {
        row[f"{slot}_text"]: row[f"{slot}_tokens"]
        for row in rows for slot in ABILITIES
    }
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    vectorizer.fit(row[f"{slot}_tokens"] for row in rows for slot in ABILITIES)
    return vectorizer, tokens_by_text


def vectorize_abilities(texts):
    vectorizer, tokens_by_text = fitted_tfidf()
    tokens = [
        tokens_by_text[text] if text in tokens_by_text
        else full_preprocess(remove_numbers(remove_markup(text)))
        for text in texts
    ]
    return vectorizer.transform([*tokens, " ".join(tokens)])


def compare_vectors(abilities_a, abilities_b):
    vectors_a = vectorize_abilities([abilities_a.get(slot, "") for slot in ABILITIES])
    vectors_b = vectorize_abilities([abilities_b.get(slot, "") for slot in ABILITIES])
    # TF-IDF rows have unit length, so their dot products are cosine similarities.
    return np.clip((vectors_a @ vectors_b.T).toarray(), 0.0, 1.0)


def ability_matrix(abilities_a: dict[str, str], abilities_b: dict[str, str]):
    """Return the 5x5 similarity matrix in P, Q, W, E, R order."""
    return compare_vectors(abilities_a, abilities_b)[:-1, :-1]


def match_abilities(matrix, threshold=MATCH_THRESHOLD):
    # Zero out weak pairs before optimizing so they cannot steal a strong match.
    eligible = np.where(matrix >= threshold, matrix, 0.0)
    rows, columns = linear_sum_assignment(eligible, maximize=True)
    return [
        (ABILITIES[row], ABILITIES[column], float(matrix[row, column]))
        for row, column in zip(rows, columns)
        if eligible[row, column] > 0
    ]


def kit_similarity(abilities_a: dict[str, str], abilities_b: dict[str, str]) -> dict:
    scores = compare_vectors(abilities_a, abilities_b)
    return {
        "score": float(scores[-1, -1]),
        "matches": match_abilities(scores[:-1, :-1]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("champion_a")
    parser.add_argument("champion_b")
    args = parser.parse_args()
    abilities = load_abilities()
    try:
        name_a = resolve_champion(args.champion_a, abilities)
        name_b = resolve_champion(args.champion_b, abilities)
    except ValueError as error:
        parser.error(str(error))
    a, b = abilities[name_a], abilities[name_b]
    result = kit_similarity(a, b)
    print(f"Kit cosine: {result['score']:.3f}")
    print("     " + "  ".join(f"{slot:>5}" for slot in ABILITIES))
    for slot, scores in zip(ABILITIES, ability_matrix(a, b)):
        print(f"{slot:>3}  " + "  ".join(f"{score:5.2f}" for score in scores))
    for slot_a, slot_b, score in result["matches"]:
        print(f"\n{name_a} {slot_a} / {name_b} {slot_b}: {score:.3f}")
        print(a[slot_a])
        print(b[slot_b])
