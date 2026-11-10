"""Learn champion embeddings from mastery pools with skip-gram Word2Vec."""

import argparse
import csv
from functools import cache
from zlib import crc32

import numpy as np
from gensim.models import Word2Vec

from data_io import DATA_DIR, read_player_pools


def load_player_pools(path=DATA_DIR / "player_pools.csv"):
    # Split on pipes so a name like "Lee Sin" stays a single token.
    pools = list(read_player_pools(path).values())
    if not pools:
        raise ValueError("The player-pool CSV has no training rows.")
    return pools


def stable_hash(text):
    return crc32(text.encode("utf-8"))


def train_word2vec(pools, seed=42):
    return Word2Vec(
        sentences=pools,
        sg=1,  # Skip-gram predicts the other champions in the same pool.
        vector_size=48,
        window=4,
        shrink_windows=False,
        min_count=1,
        sample=0,
        epochs=35,
        workers=1,
        seed=seed,
        hashfxn=stable_hash,
    ).wv


def rank_word2vec(vectors, mains, k=5):
    # Average the original embeddings, then compute cosine to each unit vector.
    query = np.mean(vectors[list(mains)], axis=0)
    norm = np.linalg.norm(query)
    scores = vectors.get_normed_vectors() @ (query / norm) if norm else np.zeros(len(vectors))
    candidates = [
        (name, float(score))
        for name, score in zip(vectors.index_to_key, np.clip(scores, -1, 1))
        if name not in mains
    ]
    return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:k]


@cache
def default_model():
    return train_word2vec(load_player_pools())


def recommend(mains: list[str], k: int = 5) -> list[tuple[str, float]]:
    """Return up to k unseen champions with their raw cosine scores."""
    if k < 0:
        raise ValueError("k must be zero or greater.")
    cleaned = [name.strip() for name in mains if name.strip()]
    if not cleaned:
        raise ValueError("Enter at least one champion.")
    vectors = default_model()
    canonical = {name.casefold(): name for name in vectors.index_to_key}
    unknown = sorted({name for name in cleaned if name.casefold() not in canonical})
    if unknown:
        raise ValueError(f"Unknown or untrained champion: {', '.join(unknown)}")
    mains = list(dict.fromkeys(canonical[name.casefold()] for name in cleaned))
    return rank_word2vec(vectors, mains, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mains", nargs="+", help="Quote champion names with spaces")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    try:
        for name, score in recommend(args.mains, args.k):
            print(f"{name}: {score:.3f}")
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))
