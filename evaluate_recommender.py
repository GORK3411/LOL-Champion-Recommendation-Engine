"""Evaluate Word2Vec, popularity, and co-occurrence on held-out player pools."""

import argparse
from collections import Counter, defaultdict
from random import Random

from recommender import load_player_pools, rank_word2vec, train_word2vec

CUTOFFS = (1, 5, 10)
RARE_LIMIT = 100


def split_players(pools, seed):
    random = Random(seed)
    shuffled = list(pools)
    random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    training = shuffled[:split]
    test = []
    for pool in shuffled[split:]:
        index = random.randrange(len(pool))
        test.append((pool[:index] + pool[index + 1:], pool[index]))
    return training, test


def train_baselines(pools):
    popularity = Counter(champion for pool in pools for champion in pool)
    pairs = defaultdict(Counter)
    for pool in pools:
        for main in pool:
            pairs[main].update(candidate for candidate in pool if candidate != main)
    return popularity, pairs


def rank_cooccurrence(mains, popularity, pairs):
    scores = Counter()
    for main in mains:
        for candidate, count in pairs.get(main, {}).items():
            scores[candidate] += count / popularity[main]
    return sorted(popularity, key=lambda name: (-scores[name], -popularity[name], name))


def evaluate(pools, seed=42):
    training, test = split_players(pools, seed)
    if not training or not test:
        raise ValueError("Need at least one training pool and one test pool.")
    popularity, pairs = train_baselines(training)
    popular = sorted(popularity, key=lambda name: (-popularity[name], name))
    vectors = train_word2vec(training, seed)
    counts = Counter()
    hits = Counter()
    for mains, target in test:
        cohort = "rare" if popularity[target] < RARE_LIMIT else "common"
        known = [name for name in mains if name in vectors]
        ranked = {
            "Popularity": popular,
            "Co-occurrence": rank_cooccurrence(mains, popularity, pairs),
            "Word2Vec": [name for name, _ in rank_word2vec(vectors, known, max(CUTOFFS))]
            if known else [],
        }
        for group in ("all", cohort):
            counts[group] += 1
            for method, names in ranked.items():
                candidates = [name for name in names if name not in mains]
                for k in CUTOFFS:
                    hits[method, group, k] += target in candidates[:k]
    return len(training), counts, hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    training, counts, hits = evaluate(load_player_pools(), args.seed)
    print(f"Seed {args.seed} | train {training:,} | test {counts['all']:,}")
    print(f"Rare means fewer than {RARE_LIMIT} training pools.")
    print(f"{'Method':<16} {'Group':<8} {'N':>5}  {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8}")
    for method in ("Popularity", "Co-occurrence", "Word2Vec"):
        for group in ("all", "rare", "common"):
            total = counts[group]
            rates = [f"{hits[method, group, k] / total:8.2%}" if total else "     n/a"
                     for k in CUTOFFS]
            print(f"{method:<16} {group:<8} {total:>5}  " + " ".join(rates))


if __name__ == "__main__":
    main()
