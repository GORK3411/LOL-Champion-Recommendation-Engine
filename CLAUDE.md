# LoL Champion Recommender – NLP Course Project

Team project (5 members) for an NLP course in an AI master's.

## What the project is about

We build a system that **recommends new League of Legends champions** to a player based on the champions they already main, then **explains why** each champion was recommended, using the champions' **ability descriptions**.

### Key game concepts

- **Champion abilities:** each champion has **5 abilities**: a Passive (P) and four spells (Q, W, E, R). Each ability has a text description. This is our text data.
- **Mastery points:** players earn mastery points on a champion by playing it. **High mastery points on a champion mean the player has played that champion a lot**, so mastery tells us a player's mains.

### The project does three things

1. **Recommend** champions based on the player's current mains (recommendation engine).
2. **Compare** each recommended champion with the player's mains to show how similar their kits are (kit similarity).
3. **Tag** each champion with labels to show what they have in common (tagging).

Part 1 decides **what** to recommend. Parts 2 and 3 explain **why**.

### Example of the final output (numbers are illustrative)

```
Input: my mains are Thresh, Nami, Janna

Recommended: Blitzcrank
Kit similarity with your Thresh: 81%
  • Thresh Q ↔ Blitzcrank Q: both pull an enemy toward them
  • Thresh E ↔ Blitzcrank E: both displace an enemy (knock-back / knock-up)
Shared tags: hook, crowd control, engage
```

### How everything connects

```
User enters mains (Thresh, Nami, Janna)
        │
        ▼
PART 1  Recommendation engine  ──►  top-k champions (Blitzcrank, Lulu, ...)
        │
        ▼
PART 2  Kit similarity         ──►  score + matched abilities
PART 3  Tagging                ──►  shared tags
        │
        ▼
Demo app shows recommendations + explanations
```

---

## Part 1 – Recommendation engine

**Uses:** player data (who plays what). This part does **not** use text.

**Goal:** given a player's mains, find champions they are likely to enjoy.

**How:** the Word2Vec recommendation method from the course. Each player becomes a "sentence" of champion names, ordered by mastery points, and Word2Vec is trained on all these sentences. Champions often played by the same players end up close together, like words that appear in similar contexts. To recommend, rank champions by cosine similarity to the user's mains (e.g. to the average of their vectors), excluding champions the user already plays.

```
Player 1: Thresh  Nami  Janna  Lulu  Blitzcrank
Player 2: Yasuo  Yone  Zed  Talon  Akali
Player 3: Thresh  Blitzcrank  Pyke  Nautilus
...      (thousands of players = our corpus)
```

- **Data:** Riot Games API (champion mastery endpoint)
- **Evaluation:** hide one of a player's mains and check whether it appears in the top-k (hit rate@k). Compare against a popularity baseline (always recommend the most-played champions).

**Interface:**

```python
def recommend(mains: list[str], k: int = 5) -> list[tuple[str, float]]:
    # mains: the user's mains (champion names)
    # k: number of champions to recommend
```

```python
recommend(["Thresh", "Nami", "Janna"], k=5)
# -> [("Blitzcrank", 0.82), ("Lulu", 0.79), ("Pyke", 0.75), ...]
```

---

## Part 2 – Kit similarity

**Uses:** ability descriptions (text).

**Goal:** compare a recommended champion with one of the player's mains and show **how similar** their kits are and **which abilities match**.

**How:** the team will **decide later** which technique to use. Recommended options: TF-IDF and Word2Vec are covered in the course; Sentence-BERT is built on BERT, which the course introduces.

- **Preprocessing:** remove numbers, scaling values and markup; replace champion names with "the champion".
- **Overall score:** turn each champion's full kit into a vector and compute cosine similarity. Candidate methods: **TF-IDF**, **Word2Vec**, **Sentence-BERT**.
- **Ability matching:** embed each ability (P, Q, W, E, R) separately, build a 5×5 similarity matrix, and find the best one-to-one pairing (e.g. with the **Hungarian algorithm**, `scipy.optimize.linear_sum_assignment`). Show only pairs above a threshold (e.g. > 0.5).

**Interface:**

```python
def kit_similarity(abilities_a: dict[str, str], abilities_b: dict[str, str]) -> dict:
    # abilities_a: ability descriptions of one of the user's mains
    # abilities_b: ability descriptions of a recommended champion
    # each dict maps an ability (P, Q, W, E, R) to its description
```

```python
kit_similarity(abilities["Thresh"], abilities["Blitzcrank"])
# abilities["Thresh"] = {"P": "...", "Q": "Thresh throws his scythe, pulling the first enemy hit toward him", ...}
# -> {"score": 0.81, "matches": [("Thresh Q", "Blitzcrank Q", 0.86), ("Thresh E", "Blitzcrank E", 0.62)]}
```

Output: `score: float` (0–1) and `matches: list[tuple[str, str, float]]`.

5×5 ability similarity matrix, Thresh vs Blitzcrank (illustrative values):

```
              Blitz P   Blitz Q   Blitz W   Blitz E   Blitz R
Thresh P       0.21      0.10      0.15      0.12      0.18
Thresh Q       0.12      0.86      0.09      0.31      0.14
Thresh W       0.25      0.08      0.22      0.10      0.17
Thresh E       0.11      0.35      0.13      0.62      0.20
Thresh R       0.14      0.18      0.16      0.21      0.40
```

---

## Part 3 – Tagging

**Uses:** ability descriptions (text).

**Goal:** give each champion short, readable labels describing what its kit does. The **shared tags** show what the recommended champion has in common with the player's mains.

**How** (methods can be compared):

- **Keyword dictionary:** "stun", "root", "knock up" → _crowd control_; "dash", "blink", "leap" → _mobility_.
- **TF-IDF keywords:** the most important words in each champion's text.
- **Zero-shot classification:** a transformer model picks which tags from our list fit each ability (see the appendix).

**Interface:**

```python
def get_tags(abilities: dict[str, str], labels: list[str]) -> list[str]:
    # abilities: ability (P, Q, W, E, R) -> description of one champion
    # labels: the list of possible tags we define

def shared_tags(tags_a: list[str], tags_b: list[str]) -> list[str]:
    # tags_a, tags_b: tags of two champions (output of get_tags)
```

```python
get_tags(abilities["Thresh"], labels=["crowd control", "mobility", "shield", "heal", ...])
# -> ["hook", "crowd control", "shield", "engage"]
shared_tags(tags["Thresh"], tags["Blitzcrank"])
# -> ["hook", "crowd control", "engage"]
```

The tags for all champions are stored in a `dict[str, list[str]]`.

### Kit similarity vs. Tagging

|          | Kit similarity (Part 2)                                | Tagging (Part 3)                              |
| -------- | ------------------------------------------------------ | --------------------------------------------- |
| Works on | A **pair** of champions                                | **One** champion at a time                    |
| Output   | A number + matched abilities                           | A list of words                               |
| Method   | Embeddings + cosine similarity                         | Keyword dictionary / zero-shot classification |
| Answers  | "_How_ similar are they, and _which_ abilities match?" | "_What_ does this champion do?"               |

**We need both:** a score alone isn't readable ("81% similar… but why?"), and tags alone are too vague (many champions have _crowd control_). Together: "81% similar: both **hook** and **engage** (Q ↔ Q)".

---

## Data

| Data                            | Source                                                            | Output file           |
| ------------------------------- | ----------------------------------------------------------------- | --------------------- |
| Player mastery (who plays what) | Riot Games API (champion mastery endpoint; needs a developer key) | `player_pools.csv`    |
| Champion ability descriptions   | Riot Data Dragon (official static game data)                      | `champions_clean.csv` |

**Text preprocessing:** cleaning, tokenization, lowercasing, stopword removal, lemmatization, and a synonym dictionary that maps game terms to shared concepts (stun / root / knock-up → _crowd control_).

**Champion IDs:** use the same champion names or IDs in every module. Data Dragon uses internal IDs that can differ from display names (e.g. `MonkeyKing` = Wukong).

---

## How the work is split

| #   | Role                                    | Tasks                                                                                                                                                                                             | Output                                    |
| --- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1   | **Player data + champion descriptions** | Collect and clean the data for all parts: player mastery data (Riot API) → each player's champion "sentence"; ability descriptions (Data Dragon) → preprocessing                                  | `player_pools.csv`, `champions_clean.csv` |
| 2   | **Recommender (Part 1)**                | Train Word2Vec on player sentences, generate top-k recommendations, evaluate with hit rate@k vs. a popularity baseline                                                                            | `recommend(mains) → top-k`                |
| 3   | **Kit similarity (Part 2)**             | Choose and implement the similarity technique (TF-IDF / Word2Vec / Sentence-BERT); ability-to-ability matching                                                                                    | `kit_similarity(A, B) → score + matches`  |
| 4   | **Tags (Part 3)**                       | Extract tags for each champion from its ability descriptions; compute shared tags                                                                                                                 | `champion → tags`                         |
| 5   | **Integration**                         | Connect all modules into one system (same file formats and champion IDs); build the demo app (e.g. Streamlit): the user types their mains and sees recommendations + kit similarity + shared tags | Working demo                              |
| All | **Report + PowerPoint**                 | Everyone works on both; each person explains their own part                                                                                                                                       | Report + PowerPoint                       |

### Order of work

1. **Phase 1:** Player data + champion descriptions
2. **Phase 2:** Parts 1, 2 and 3 (Recommender, Kit similarity, Tags)
3. **Phase 3:** Integration
4. **Phase 4:** Report + PowerPoint

---

## Notes for Claude Code

- The function signatures above are the **contract between team members**. Don't change them without flagging it, since other modules depend on them.
- The technique for Part 2 is **not decided yet**. When asked, compare options rather than assuming one.
- All numbers in this file (scores, similarities) are **illustrative**, not real results.
- Player data comes from the **Riot Games API** only.
- This is an NLP course project: prefer techniques from the course (text preprocessing, Bag of Words, TF-IDF, Word2Vec, BERT / Transformers) and explain any technique that goes beyond it.
- Keep code readable and explainable: each team member must present their own part.

---

## Appendix A – Hungarian algorithm (Part 2, ability matching)

It finds the **best one-to-one pairing** between two groups: each row is matched with exactly one column, and the total score is as good as possible. Here it pairs the abilities of two champions. It is not covered in the course.

**Why not greedy?** An early "best" choice can block better pairs later. Example (similarity × 10):

|              | Blitz Q | Blitz W | Blitz E |
| ------------ | ------- | ------- | ------- |
| **Thresh Q** | 9       | 8       | 1       |
| **Thresh W** | 8       | 2       | 1       |
| **Thresh E** | 1       | 1       | 5       |

- **Greedy:** Q↔Q (9) first leaves Thresh W only Blitz W (2); with E↔E (5), the total is **16**.
- **Optimal:** Q↔W (8) + W↔Q (8) + E↔E (5) = **21**.

**Steps.** The algorithm minimizes cost, so first convert: cost = 10 − similarity → `1 2 9 / 2 8 9 / 9 9 5`.

1. **Row reduction:** subtract each row's minimum → `0 1 8 / 0 6 7 / 4 4 0`
2. **Column reduction:** subtract each column's minimum → `0 0 8 / 0 5 7 / 4 3 0`
3. **Cover all zeros with the fewest lines.** If the number of lines = n (here 3), an optimal assignment exists among the zeros: go to step 5.
4. **If fewer than n lines:** subtract the smallest uncovered number from every uncovered cell, add it to cells covered twice, and repeat step 3.
5. **Pick one zero per row and per column:** TW↔BQ, TQ↔BW, TE↔BE → total similarity 8 + 8 + 5 = **21**.

Subtracting a constant from a whole row or column doesn't change _which_ pairing is best, so the algorithm simplifies the matrix until the best pairing shows up as zeros. In graph terms, it finds the maximum-weight perfect matching in a bipartite graph (the assignment problem).

```python
from scipy.optimize import linear_sum_assignment
rows, cols = linear_sum_assignment(-sim)   # negative because we maximize similarity
```

## Appendix B – Zero-shot classification (Part 3, tagging)

A **pre-trained transformer** sorts text into labels **we define**, **without training data**. It gives each label a score (0–1), and we keep the labels above a threshold. It is not covered in the course, but it runs on the Hugging Face Transformers library, which the course introduces.

**How it works:** the model was trained on Natural Language Inference (does one sentence imply another?). For each label it checks a hypothesis:

- Text: "Thresh throws his scythe, pulling the first enemy hit toward him."
- Hypothesis: "This text is about **crowd control**."
- How likely the text implies the hypothesis = that label's score.

```python
from transformers import pipeline

clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "Thresh throws his scythe, pulling the first enemy hit toward him."
labels = ["crowd control", "mobility", "shield", "heal", "damage"]

res = clf(text, labels, multi_label=True)
tags = [l for l, s in zip(res["labels"], res["scores"]) if s > 0.5]
```

Possible result (illustrative): crowd control 0.94 ✓, damage 0.41, mobility 0.18, shield 0.03, heal 0.02 → Thresh Q gets the tag **crowd control**.

**Tips:**

- Use `multi_label=True`, because one ability can have several tags.
- Classify **each ability separately**, then combine the tags for the champion.
- Descriptive labels ("crowd control (stun, root, slow)") work better than abbreviations ("CC").
- Tune the threshold (start at 0.5) by checking a few champions manually.

---

## Learning resources

The first four are the resources linked in the course PDF.

- **TF-IDF + Word2Vec:** [An Intuitive Understanding of Word Embeddings: From Count Vectors to Word2Vec](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
- **Word2Vec recommender:** [Building a Recommendation System using Word2vec: A Unique Tutorial with Case Study in Python](https://www.analyticsvidhya.com/blog/2019/07/how-to-build-recommendation-system-word2vec-python/)
- **Transformers / BERT:** [Introduction to PyTorch-Transformers](https://www.analyticsvidhya.com/blog/2019/07/pytorch-transformers-nlp-python/)
- **Preprocessing:** [Steps for effective text data cleaning (with case study using Python)](https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/)
- **Sentence-BERT** (not in the course material): [official documentation](https://www.sbert.net/)
