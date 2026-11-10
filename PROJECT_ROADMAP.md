# League of Legends Champion Recommender

Use this file as the handoff for the rest of the NLP course project. The original team brief and function contracts are in [CLAUDE.md](CLAUDE.md). The Windows setup and commands are in [README.md](README.md).

**Goal.** A player selects champions they already play. The app ranks new champions from player behavior and explains each result with actual ability text, matched mechanics, and shared tags. A classmate should be able to follow how every score and label was produced.

## 1. Audit and freeze the data

- [x] Use the committed snapshot as the input for local training and evaluation. It contains 10,677 unique player pools and 173 champions with five ability descriptions each.
- [x] Validate unique player and champion IDs, 2–5 distinct champions per pool, catalog joins, raw-to-light text agreement, and whole-kit column construction. Run `python validate_data.py`.
- [x] Remove the Riot API key fallback. Read the key from the environment only when collecting live player data.
- [x] Bound timeouts and temporary retries. Stop on authentication failures and preserve existing rows when resuming the player collector.
- [x] Complete a description download and preprocessing pass before replacing files. Expose a patch argument and an offline cleaning command.
- [x] Reject duplicate or malformed player records before training. Verify that the collector can resume from a file with no final newline and that a blocked cleaned CSV preserves the previous raw CSV.
- [ ] Have the key owner revoke or rotate the old key if it is still active. Coordinate any Git history cleanup with the repository owner.
- [ ] Recover or record the exact Data Dragon version and collection date if available. The frozen CSVs do not store them. Record the patch and collection settings alongside any future refresh.

### What to study

- [champion_collection.py](champion_collection.py) turns champion mastery rows into one pipe-separated pool per player. This is behavior data from an EUW ranked sample spanning Bronze to Platinum. The 30,000-mastery cutoff and top-five limit shape what the model learns.
- [champion_descriptions.py](champion_descriptions.py) produces two versions of every ability. Light text keeps sentence structure. Tokens remove common words, lemmatize, and consolidate mechanics such as `grab` and `drag` into `pull`. Compare Thresh Q and Blitzcrank Q in [the cleaned CSV](data/champions_clean.csv).
- Preserve the four committed CSVs as the reference snapshot. Run the two collectors one at a time and keep a copy before refreshing them. Use the validator to catch discrepancies after a refresh. If the raw and cleaned descriptions get out of sync, rerun `python champion_descriptions.py --clean-only`.

## 2. Train and evaluate the recommender

- [x] Implement `recommend(mains, k=5)` using Word2Vec skip-gram and cosine similarity.
- [x] Support multiword names, surrounding whitespace and case differences. Deduplicate inputs, reject unknown names, and exclude the selected mains from results.
- [x] Evaluate Word2Vec, popularity and co-occurrence with one hidden champion per held-out player. Compute all models and baselines from training pools only. Report Hit@1, Hit@5 and Hit@10, including rare and common target groups.
- [x] Repeat the same evaluation on three fixed seeds and record the results below.
- [ ] Have a teammate review the protocol. If tuning settings, choose them on a validation split and keep a final test untouched.
- [ ] Add coverage and failure examples for rare champions. Ask players to judge usefulness if the report makes claims about new champion adoption or enjoyment.

### What to study

1. [recommender.py](recommender.py) splits on `|`, keeping `Lee Sin` as one token. Skip-gram predicts the other champions in a pool. A window of four covers every position in pools of up to five.
2. `rank_word2vec` averages the original embeddings of the selected mains. It ranks other champion embeddings by cosine similarity to that average. The scores describe closeness in the learned space.
3. [evaluate_recommender.py](evaluate_recommender.py) holds out players before training. The popularity baseline counts champions. The co-occurrence baseline sums the estimated likelihood of a candidate appearing with each visible main. A hit means the hidden champion appears in the top-k results.

### Measured Hit@5

Each run used 8,541 training pools and 2,136 held-out pools. Rare targets appeared in fewer than 100 training pools.

| Seed | Popularity | Co-occurrence | Word2Vec | Rare targets | Word2Vec on rare targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 15.22% | 26.54% | 9.50% | 178 | 10.11% |
| 43 | 14.56% | 25.61% | 8.24% | 142 | 7.04% |
| 44 | 16.06% | 28.04% | 9.46% | 166 | 7.83% |

Co-occurrence led overall on all three splits. Both baselines scored 0% on rare targets at five in these runs. Keep Word2Vec as the course method and show the stronger overall baseline in the demo. This evaluation measures recovery of an existing main from the same snapshot. It provides no observation of future adoption.

## 3. Implement the explanations

- [x] Use TF-IDF on the frozen ability tokens. Compute whole-kit cosine and build the 5×5 matrix in passive, Q, W, E, R order.
- [x] Pair abilities with one-to-one maximum-score assignment. Ignore candidates below the current 0.20 threshold before assignment, so weak pairs cannot block a strong pair.
- [x] Implement the fixed tag vocabulary and expose the matched ability slot and phrase for each tag. Keep shield and shield break separate.
- [x] Check descriptions that mention immunity or self-applied effects. Verify that Master Yi's slow immunity and Bel'Veth's self-root do not count as enemy crowd control.
- [x] Add a reproducible comparison harness for TF-IDF and MiniLM using a 24-pair starter set with positive and confusable negative examples.
- [ ] Have two teammates independently label a larger set of pairs and agree on the annotation rubric. Reserve separate examples to choose the match threshold. Document systematic false positives and false negatives for both methods.
- [ ] Review the tag taxonomy with players. Check variants such as ally movement, pulling yourself versus pulling an opponent, breaking versus granting a shield, and removing crowd control versus applying it.

### What to study

1. [kit_similarity.py](kit_similarity.py) gives weight to terms that distinguish abilities across the corpus. With L2-normalized vectors, a dot product gives cosine similarity. The same fitted vocabulary represents abilities and whole kits.
2. Run `python kit_similarity.py Thresh Blitzcrank` to see the actual matrix and matching text. Q can match E when the wording supports it. A high score can still reflect shared wording with different mechanics.
3. [tagging.py](tagging.py) matches readable rules against natural text. `tag_evidence` traces each label to one or more slots. `get_tags` filters to the requested labels and `shared_tags` computes the intersection in stable order.
4. [evaluate_similarity.py](evaluate_similarity.py) measures how often a labeled positive pair scores above a labeled negative pair. The examples are a small, hand-labeled teaching set. Treat the comparison as diagnostic evidence.

### Measured similarity examples

- On 24 hand-labeled pairs, TF-IDF ranked positives above negatives **83.3%** of the time. MiniLM scored **66.0%** by the same measure. The pair set is small and intentionally includes confusing examples.
- Thresh Q and Blitzcrank Q scored **0.442** with TF-IDF and passed the match threshold. Their whole-kit cosine was **0.193**.
- The **0.20** threshold accepted 5 of the 12 labeled positive pairs and 1 of the 12 negatives when scored individually. It missed Nami W and Soraka W even though both heal allies. It accepted Pyke Q and Amumu Q because both discuss pulling, even though they pull different targets. Keep the ability text visible so the user can check the explanation.
- The default matcher stays with TF-IDF for a readable, offline implementation. The MiniLM run is available for comparison.

## Verification before integration

- [x] Validate the committed snapshot and regenerate the cleaned CSV in a temporary copy. All 173 regenerated rows matched the committed values.
- [x] Pass the local unit tests covering collector retries and restart behavior, snapshot joins, recommendations, ability assignment, and tag evidence.
- [x] Run the recommender and both similarity evaluators in the pinned Windows Python 3.12 environment. Live Riot collection still needs verification with a valid key if the team refreshes the player data.
- [x] Smoke-test public Data Dragon downloads with Annie, Wukong and Lee Sin in a temporary directory. This covers a direct champion ID, a different internal ID, and a multiword display name.

## 4. Integrate and build the demo UI

- [ ] Build a local app with one main recommendation screen. Streamlit is the simplest option if the team wants to stay in Python.
- [ ] Use a searchable multiselect of champion names, clear input validation, a top-k control, and a one-click sample using Thresh, Nami and Janna.
- [ ] Give users a way to compare Word2Vec and co-occurrence rankings on the same input. Use the recorded benchmark when deciding the demo default.
- [ ] Keep results in a consistent order and label the **ranking cosine** and **kit cosine** separately. Never present either as a probability that a player will enjoy the champion.
- [ ] For each result, show the input champion used for the comparison, the accepted ability pairs, their original text, and shared tags with evidence. If there are no strong matches, show a clear empty state.
- [ ] Add an expandable **How this works** section. Expose the selected champion vectors or a small readable projection, the query average, the ranked cosine scores, and the color-coded 5×5 matrix with the assigned cells outlined. Show raw text, light text, and normalized tokens for one real ability.
- [ ] Make the interface presentable on a projector and usable on a narrow screen. Use a clear reading order, strong text contrast, keyboard-accessible controls, labeled scores, and text labels alongside color. Use the same visual language for every result.
- [ ] Cache the recommender and corpus for the session. Confirm that an existing data snapshot supports the entire demo offline without a Riot key.
- [ ] Test empty and duplicate selections, mixed roles, multiword names, no accepted matches, absent data, an out-of-vocabulary input, and a fresh launch on the classroom Windows machine.
- [ ] Make a simple launch script and rehearsal instructions once the UI works. Verify the packaged or extracted version separately if one is produced.

**Acceptance.** A classmate can open the app, select mains, get recommendations, and explain one recommendation using its rank, a real ability match and a shared tag. The presenter can open the underlying math without switching applications.

## 5. Write the report and prepare the presentation

- [ ] Document data provenance, sample limitations, preprocessing, algorithms, training split, and reproducible commands.
- [ ] Include a real baseline comparison, a rare-champion breakdown, a 5×5 matrix, a tag evidence example, and one failure case. Replace all illustrative design scores with measured values.
- [ ] Give each team member a section and a short explanation they can deliver in their own words.
- [ ] Rehearse the live flow and keep screenshots or terminal output from a verified run as a fallback.
- [ ] Check that a fresh environment can install dependencies and run the validator, tests, evaluation and demo from the README.

## Resume as a goal

> Continue the champion recommender using PROJECT_ROADMAP.md. Verify milestones 1–3, then complete milestones 4–5 and update each checkbox with evidence. Keep the Python readable and the algorithm steps visible in the demo. Preserve the checked-in snapshot. Start with the smallest end-to-end flow, then polish and test it on Windows.
