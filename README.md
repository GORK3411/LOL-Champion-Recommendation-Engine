# LOL-Champion-Recommendation-Engine

Recommends League of Legends champions from player mastery pools and compares their ability descriptions. The data, recommender, and explanation modules run locally. See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for the study guide and remaining demo work, and [CLAUDE.md](CLAUDE.md) for the original team brief.

## Setup

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python validate_data.py
& $python -m unittest discover -s tests -v
```

The examples below use the committed snapshot and do not need a Riot key. Install the spaCy English model only to regenerate tokens or compare ability text outside the snapshot.

```powershell
& $python -m spacy download en_core_web_sm
```

## Run the NLP modules

```powershell
& $python recommender.py Thresh Nami Janna -k 5
& $python evaluate_recommender.py --seed 42
& $python kit_similarity.py Thresh Blitzcrank
& $python tagging.py Rell
& $python evaluate_similarity.py
```

Quote names that contain spaces, for example `"Lee Sin"`. All three commands accept case differences and surrounding spaces. Ranking scores are raw cosine similarities. The kit command prints the whole-kit score, the 5×5 ability matrix and the accepted matches. The tag command prints the phrases that triggered each label.

For the optional MiniLM comparison, the first run downloads model weights into a local cache.

```powershell
& $python -m pip install fastembed==0.9.1
& $python evaluate_similarity.py --semantic
```

## Data collection scripts

### `champion_collection.py`: player champion pools (Part 1 data)

Collects each player's most played champions from the Riot Games API.

1. **Get a Riot API key** at <https://developer.riotgames.com>. Development keys expire every 24 hours.
2. **Set the key** as an environment variable (don't commit a real key to the repo):
   ```powershell
   $env:RIOT_API_KEY = "RGAPI-..."      # PowerShell
   ```
   ```bash
   export RIOT_API_KEY="RGAPI-..."      # bash
   ```
3. **Run it:**
   ```bash
   python champion_collection.py --pages 10 --top 5
   ```

What it does:

- Downloads the champion list from Data Dragon into `data/champions_id.csv` (only if the file doesn't exist yet).
- Gets players from the EUW ranked solo ladder (tiers Bronze, Silver, Gold and Platinum, divisions I–IV). By default it reads 10 pages per tier/division (`main(10)` at the bottom of the file).
- For each player, gets their top 5 champions by mastery points and keeps only the champions with at least 30,000 points (`MIN_MASTERY_POINTS`).
- Saves players with at least 2 champions left (`MIN_CHAMPIONS`) to `data/player_pools.csv`.

Notes:

- The script **appends** to `data/player_pools.csv` and skips players already saved, so you can run it again (for example with more pages) to add players without fetching existing ones again.
- Temporary network errors, rate limits and server errors get bounded retries. Authentication errors stop the run. Progress is flushed every 100 pending players.
- The collector and evaluator reject malformed player-pool files and duplicate player IDs. Keep the `puuid,champions` header in that order.
- Use `--pages` and `--top` to control the run. Run one collector at a time so a second process does not append the same players.

### `champion_descriptions.py`: ability descriptions (Parts 2 and 3 data)

Downloads every champion's ability descriptions from Data Dragon and preprocesses them. No API key is needed.

```bash
python champion_descriptions.py
```

It needs `data/champions_id.csv`, so run `champion_collection.py` first (or at least call its `get_champions_by_id()` function).

What it does:

1. Downloads the passive and four spell descriptions of each catalog champion → `data/champion_descriptions.csv`. It uses the latest Data Dragon version by default. Use `--patch VERSION` to specify a version and `--clean-only` to reuse the raw CSV.
   The refresh stages both CSVs before publishing and restores the previous raw file if the cleaned file cannot be replaced.
2. Cleans them in two versions → `data/champions_clean.csv`:
   - **Light cleaning** (`*_text`): removes HTML tags, template placeholders (`{{ e1 }}`, `@Effect1Amount@`) and numbers, and replaces the champion's name with "the champion". The text stays as natural sentences, for Sentence-BERT or zero-shot classification.
   - **Full preprocessing** (`*_tokens`): starts from the light-cleaned text, then lowercases, merges multi-word game terms into one token ("knocked up" → `knockup`, "attack speed" → `attackspeed`), tokenizes and lemmatizes with spaCy, removes standard and custom stopwords, and maps synonyms to a shared concept ("blink", "leap" → `dash`). For TF-IDF, Word2Vec or keyword tagging.

The dictionaries used for preprocessing (`PHRASES`, `CUSTOM_STOPWORDS`, `SYNONYMS`, `NAME_PARTS_EXCLUDE`) are at the top of the file and can be edited. To re-run only the cleaning without downloading again:

```python
from champion_descriptions import clean
clean("data/champion_descriptions.csv")
```

## CSV files (`data/`)

Champions are identified by their **display name** (e.g. `Wukong`, `Lee Sin`, `Kai'Sa`) in every file, so the files can be joined on the champion name.

### `champions_id.csv`

Created by `champion_collection.py`. One row per champion (173 champions).

| Column | Description                                      | Example |
| ------ | ------------------------------------------------ | ------- |
| `id`   | Riot numeric champion id (used by the mastery API) | `1`     |
| `name` | Champion display name                            | `Annie` |

### `player_pools.csv`

Created by `champion_collection.py`. One row per player (about 10,700 players). This is the Word2Vec corpus for Part 1.

| Column      | Description                                                                                                      | Example                                   |
| ----------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `puuid`     | Riot's unique player id                                                                                          | `I8mJ1Yiv...`                             |
| `champions` | The player's top champions, **ordered by mastery points (highest first)**, separated by `\|` (between 2 and 5 champions) | `Renekton\|Kha'Zix\|Ezreal\|Gnar\|Lee Sin` |

Champions are separated by `|` and not by spaces because some names contain spaces. To get a player "sentence": `row["champions"].split("|")`.

### `champion_descriptions.csv`

Created by `champion_descriptions.py`. One row per champion, with the **raw** Data Dragon descriptions (they still contain HTML tags like `<br>` and the champion's name).

| Column     | Description                    |
| ---------- | ------------------------------ |
| `champion` | Champion display name          |
| `P`        | Passive description (raw)      |
| `Q`        | Q spell description (raw)      |
| `W`        | W spell description (raw)      |
| `E`        | E spell description (raw)      |
| `R`        | R (ultimate) description (raw) |

### `champions_clean.csv`

Created by `champion_descriptions.py`. One row per champion, with the preprocessed descriptions. This is the input for Parts 2 and 3.

| Column                        | Description                                                                   | Example (Annie P)                                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `champion`                    | Champion display name                                                         | `Annie`                                                                                                              |
| `P_text` … `R_text`           | Light-cleaned description of each ability (natural sentences)                 | `After casting spells, the champion's next offensive spell will stun the target. The champion begins the game ...`    |
| `P_tokens` … `R_tokens`       | Fully preprocessed tokens of each ability, separated by spaces                | `spell offensive spell stun begin game respawn pyromania available`                                                  |
| `kit_text`                    | The 5 `*_text` columns joined (whole kit)                                     |                                                                                                                      |
| `kit_tokens`                  | The 5 `*_tokens` columns joined (whole kit)                                   |                                                                                                                      |

Columns in order: `champion, P_text, P_tokens, Q_text, Q_tokens, W_text, W_tokens, E_text, E_tokens, R_text, R_tokens, kit_text, kit_tokens`.

Loading the abilities in the format used by `kit_similarity` and `get_tags`:

```python
import pandas as pd

df = pd.read_csv("data/champions_clean.csv").fillna("")
abilities = {
    row["champion"]: {x: row[f"{x}_text"] for x in ["P", "Q", "W", "E", "R"]}
    for _, row in df.iterrows()
}
abilities["Thresh"]["Q"]
```
