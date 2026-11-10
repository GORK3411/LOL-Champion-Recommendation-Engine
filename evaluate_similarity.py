"""Compare TF-IDF and optional MiniLM on a small hand-labeled ability check."""

import argparse

import numpy as np

from data_io import ABILITIES, load_abilities
from kit_similarity import ability_matrix

# Each positive pair has a shared concrete effect. Negatives include tempting
# overlaps, such as mentioning a shield while breaking it instead of granting it.
CASES = (
    ("Thresh", "Q", "Blitzcrank", "Q", True, "Pull an opponent"),
    ("Pyke", "Q", "Nautilus", "Q", True, "Pull an opponent"),
    ("Rell", "R", "Blitzcrank", "Q", True, "Pull enemies toward the caster"),
    ("Janna", "E", "Karma", "E", True, "Shield an ally"),
    ("Janna", "E", "Lulu", "E", True, "Shield an ally"),
    ("Nami", "W", "Soraka", "W", True, "Heal an ally"),
    ("Janna", "R", "Sona", "W", True, "Heal nearby allies"),
    ("Alistar", "Q", "Malphite", "R", True, "Knock enemies into the air"),
    ("Alistar", "W", "Tristana", "R", True, "Knock an enemy back"),
    ("Morgana", "Q", "Karma", "W", True, "Hold an enemy in place"),
    ("Pyke", "W", "Akali", "W", True, "Gain stealth"),
    ("Kayn", "Q", "Akali", "R", True, "Dash and deal damage"),
    ("Thresh", "Q", "Ezreal", "Q", False, "Enemy projectile without a pull"),
    ("Pyke", "Q", "Amumu", "Q", False, "Pulls opponent versus pulls himself"),
    ("Janna", "E", "Rell", "Q", False, "Grants versus breaks a shield"),
    ("Karma", "E", "Blitzcrank", "R", False, "Grants versus removes shields"),
    ("Nami", "W", "Karthus", "Q", False, "Healing allies versus area damage"),
    ("Soraka", "W", "Lulu", "E", False, "Restores health versus shields"),
    ("Janna", "E", "Sivir", "R", False, "Shields versus speeds up allies"),
    ("Alistar", "Q", "Karthus", "Q", False, "Nearby damage without knockup"),
    ("Morgana", "Q", "Lux", "R", False, "Root versus damage beam"),
    ("Pyke", "W", "Sona", "E", False, "Move speed overlap without stealth"),
    ("Kayn", "Q", "Kayn", "W", False, "Dash versus slow"),
    ("Akali", "W", "Soraka", "Q", False, "Stealth versus slow and self-heal"),
)


def semantic_embeddings(abilities):
    from fastembed import TextEmbedding

    keys = list(dict.fromkeys(
        key for a, slot_a, b, slot_b, _, _ in CASES for key in ((a, slot_a), (b, slot_b))
    ))
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=2)
    return dict(zip(keys, model.embed([abilities[name][slot] for name, slot in keys])))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pairwise_accuracy(scores):
    """How often a positive pair outranks a negative pair. Ties count half."""
    positive = [score for score, label in scores if label]
    negative = [score for score, label in scores if not label]
    wins = sum(1 if a > b else 0.5 if a == b else 0 for a in positive for b in negative)
    return wins / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true", help="Also run local MiniLM embeddings")
    args = parser.parse_args()
    abilities = load_abilities()
    embeddings = semantic_embeddings(abilities) if args.semantic else {}
    tfidf, semantic = [], []
    for a, slot_a, b, slot_b, label, reason in CASES:
        matrix = ability_matrix(abilities[a], abilities[b])
        score = float(matrix[ABILITIES.index(slot_a), ABILITIES.index(slot_b)])
        tfidf.append((score, label))
        detail = f"TF-IDF {score:.3f}"
        if args.semantic:
            score = cosine(embeddings[a, slot_a], embeddings[b, slot_b])
            semantic.append((score, label))
            detail += f", MiniLM {score:.3f}"
        print(f"{a} {slot_a} / {b} {slot_b} | {detail} | "
              f"{'match' if label else 'different'} ({reason})")
    print(f"TF-IDF pairwise accuracy: {pairwise_accuracy(tfidf):.1%}")
    if args.semantic:
        print(f"MiniLM pairwise accuracy: {pairwise_accuracy(semantic):.1%}")


if __name__ == "__main__":
    main()
