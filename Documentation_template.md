# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Neural Nomads  
**Team Members:** Kumar Saurav, Nischay Kumar, Aviral Singh, Shikhar Srivastava  
**Submission Date:** 1 October 2026

---

## 1. Executive Summary

We solve the task as a two-stage entity-resolution pipeline. The first stage is a weighted
inverted-index blocker: every record contributes a set of "keys" (cleaned name words, consonant
skeletons of those words, a compressed form of the whole name, address words, numbers, and
house-number/street combinations), and each Source-1 record keeps the 10 Source-2 and 10
Source-3 records that share the rarest keys with it. The second stage is a LightGBM classifier
over 43 features: string similarities, plus "competition" features that compare a pair with
the other candidates around it. A final one-to-one assignment gives every S2/S3 record to at
most one S1 entity. The two ideas that mattered most were **consonant skeletons**, which make
romanized Devanagari/Tamil/Telugu names comparable to their English spelling, and the
**reverse-rank features**, which let the model use the fact that every S2/S3 record belongs to
exactly one business. On held-out training entities, processed exactly like the test set, the
pipeline reaches a **macro F0.5 of 0.917** (precision 0.991, recall 0.832). A fuzzy-matching
baseline built on a single blocking key scores 0.611.

---

## 2. Methodology

### 2.1 Problem Analysis

Before designing anything, we profiled the data and read a few hundred ground-truth pairs by
hand. We especially looked at pairs that a simple approach failed to find. These are the
findings that shaped the design.

**Size and shape of the data**

| Split | Source 1 | Source 2 | Source 3 | Countries |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | India, US |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | France, India, US |

- The training labels contain 7,638,365 matched pairs. Only **5.6 % of S1 entities have no
  match**. A matched S1 entity has 3.67 matches on average (median 4, maximum 11). The
  distribution is wide: 119k entities have one match, 531k have three, and a few hundred have ten
  or more.
- A single source often holds several copies of the same business: 54 % of matched S1 entities
  have two or more matches inside S2 alone. So "find the single best match per source" would
  cap recall badly. The output must be a set.
- **Every matched S2/S3 id appears under exactly one S1 entity** in the labels. We later exploit
  this as a one-to-one constraint and as a family of features.
- Matching is always within a country. Blocking by country is exact and costs no recall.
- The test set contains **France, which does not appear in training** at all. We therefore
  avoided anything that could only work for India/US (for example country-specific model
  branches). Country only enters as a single "is India" flag, and French street vocabulary is
  handled in normalization.

**Noise we observed in true matching pairs**

| Noise type | Example (S1 → S2/S3) |
|---|---|
| Zero-padded / edited house numbers | `16305 4th Street` → `016305 4th St`; `3706 Babcock St` → `7706 Babcock St` |
| Address parts reordered | `2426 625, Cedar City, UT` → `625, CEDAR CITY, UT`; state written first |
| State names vs. codes | `Washington` ↔ `WA`, `Haryana` ↔ `HR`, `Tamil Nadu` ↔ `TN` |
| Street-type and ordinal variants | `Avenue`/`Ave`/`AV`, `Second Avenue`/`2nd Ave` |
| Placeholder tokens | `<NULL>`, `N/A` inside addresses |
| Words injected into or removed from names | `Srinidhi Trading Private Limited` → `Private Srinidhi Trading Ltd`; `ESUN PRIVATE LIMITED CENTER` |
| Prefix / wrapper noise | `-- Swastik Projects Private Limited`, `Orbipyra DBA: Supreme Mobility Dynamics LLC` |
| Name written as a web handle | `Flint Group` → `FLINTGROUP.COM`, `Delta Trading Inc` → `#deltatrading` |
| Digit-for-letter substitutions | `Bangalore` → `Banga1ore`, `Kelly` → `Ke1ly` |
| Stray accents | `Payne Énterprises`, `SCI Ptit Àmicale` |
| Native-script names | `Hotel Logistics Private Limited` → `होटल लॉजिस्टिक्स प्राइवेट लिमिटेड` |
| Name replaced entirely | `KC Sterling Devices Inc` → `Pyradrexzeta` (address still matches) |

**Missing and non-Latin content** (share of records, computed after our normalization)

| Source | Name contains non-ASCII characters | Empty address | No house number |
|---|---|---|---|
| S2 India (train) | 27.9 % | 2.9 % | 9.3 % |
| S3 India (train) | 18.5 % | 3.1 % | 10.7 % |
| S2/S3 US (train) | ≈ 6.7 % | ≈ 3.6 % | ≈ 9.5 % |
| S2/S3 France (test) | ≈ 24 % | ≈ 3.0 % | ≈ 8.4 % |

S1 names are always plain ASCII. In India, the non-ASCII names are mostly full renderings in
Devanagari, Tamil or Telugu script. In France and the US, they are mostly accented Latin letters.
The Indic-script names are **phonetic spellings of the English name**, not translations
(`फूड` is "food" written in Devanagari). Transliterating them to Latin letters is therefore enough,
and no translation model is needed. This matters a lot for the compute budget.

**What a naive approach gets wrong.** As a baseline we built a pipeline that joins records on
"first name word + first number in the address" and then thresholds a weighted fuzzy score.
Only **44.8 % of the true pairs ever meet** under that key, and it scores 0.611 macro F0.5. Reading
the missed pairs showed that about 40 % of them differed only in the first name word (prefixes,
injected words, script) and 35 % only in the number (padding, typos). That told us candidate
generation was the real bottleneck, not the matcher.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (with a one-to-one assignment step)  
**Core Innovation:** Multi-key blocking weighted by inverse document frequency, run over
aggressively normalized tokens *and* their consonant skeletons. It is paired with
competition-aware features (reverse rank, gaps to the best competing pair) that teach the
classifier the one-owner-per-record structure of the data.

**Architecture**

```
                         raw TSVs (train_source1..3, test_source1..3)
                                          │
                   ┌──────────────────────▼───────────────────────┐
  Stage 1 prep.py  │ per source, per country                        │
                   │  • unidecode non-Latin text                    │
                   │  • clean name  → full / core / skeleton        │
                   │  • clean address → tokens, house no., numbers  │
                   │  • emit blocking keys (n:, k:, p:, q:, a:, d:, h:)│
                   └──────────────────────┬───────────────────────┘
                                          │  WORK/<split>_s<k>_<country>.parquet
                   ┌──────────────────────▼───────────────────────┐
  Stage 2 block.py │ inverted index over hashed keys (per country,  │
                   │ per target source). Drop keys with df > 400.   │
                   │ pair score = Σ log(N/df) over shared keys      │
                   │ keep top-10 per S1 per source                  │
                   └──────────────────────┬───────────────────────┘
                                          │  ≈20 candidates per S1
                   ┌──────────────────────▼───────────────────────┐
  Stage 3          │ 19 rapidfuzz similarities (multithreaded)      │
  features.py      │ + exact-match flags, lengths, blocking ranks   │
                   └──────────────────────┬───────────────────────┘
                   ┌──────────────────────▼───────────────────────┐
  Stage 4 train.py │ + group features (gaps / ranks vs. competitors)│
                   │ LightGBM binary classifier                     │
                   │ threshold + one-to-one tuned for macro F0.5    │
                   └──────────────────────┬───────────────────────┘
                   ┌──────────────────────▼───────────────────────┐
  Stage 5          │ p ≥ 0.70 → each S2/S3 id kept only for its best │
  predict.py       │ S1 → matching_results.tsv, candidate_pairs.tsv │
                   └──────────────────────────────────────────────┘
```

Every stage writes Parquet to a scratch folder, and each stage can resume from the previous one.
Processing goes country by country, and the expensive joins are chunked. The whole pipeline
therefore runs on an ordinary 12-core, 16 GB laptop; no dense embeddings or GPU are needed.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalization that feeds the blocker

Blocking quality depends almost entirely on how well the text is cleaned. Each record is turned
into several views.

**Names**

1. Non-ASCII text is transliterated with `unidecode` (rule-based and offline).
   `एसएस फूड प्राइवेट लिमिटेड` becomes `eses phuudd praaivett limittedd`.
2. Lower-case; `&` and `+` become `and`; web suffixes (`.com`, `.net`, `.org`, `.in`, `.co`,
   `.fr`, …) and `www.` are removed.
3. Split on anything that is not a letter or digit. A token mixing letters and digits gets
   leetspeak repair (`0→o, 1→l, 3→e, 4→a, 5→s, 7→t, @→a, $→s`), then any leftover digits are
   dropped, so `banga1ore` → `bangalore`. Pure numbers are left alone.
4. **Full name** = all tokens. **Core name** = the tokens minus a stop-list of legal forms and
   honorifics (`pvt, private, ltd, limited, llc, inc, corp, co, company, llp, plc, sarl, sas, eurl,
   sci, dba, m, s, mr, sri, shri, shree, the, and, of, de, du, la, le, …`). It also drops
   transliterated legal words, recognized by their skeletons (`prvt`, `lmtd`, …).
5. **Consonant skeleton** of each core token: `ph→f, ck→k, q→k, z→j, w→v`, collapse repeated
   letters, keep the first letter, and delete every later vowel (including `y`).

   | Word | Skeleton | Transliterated / misspelled form | Skeleton |
   |---|---|---|---|
   | hotel | `htl` | hottl | `htl` |
   | investments | `invstmnts` | innnvesttmenntts | `invstmnts` |
   | food | `fd` | phuudd | `fd` |
   | limited | `lmtd` | limittedd | `lmtd` |
   | bangalore | `bnglr` | banga1ore (after repair) | `bnglr` |

   Unidecode renders Indic vowels as doubled or extra letters. Consonants survive much better, so
   the skeleton is where the romanized name and its English source finally agree.

**Addresses**

1. Transliterate, lower-case, drop the `<null>` / `n/a` placeholders, and split into tokens.
2. Multi-word states and places are joined first (`new york→ny`, `tamil nadu→tn`,
   `andhra pradesh→ap`, `new delhi→delhi`, `hauts de france→hdf`, …).
3. Numbers lose leading zeros, including inside alphanumeric tokens (`016305a` → `16305a`).
4. Word-level maps: US/Indian/French street types (`street→st`, `avenue/av→ave`,
   `boulevard/bd→blvd`, `r→rue`, `near→nr`, `opposite/op→opp`, `nagar→ngr`, `colony→col`, …),
   ordinals (`second→2nd`), directions (`north→n`), and full state names to postal codes for
   the US and India.
5. Filler tokens are removed (`no`, `door`, `house`, `plot`, `flat`, `unit`, `apt`, …).
6. We keep the token string, the alphabetic tokens, the numeric tokens, and the **house number**
   (the first numeric token).

### 3.2 Blocking keys

Each record emits a set of typed keys (the prefix keeps key families apart):

| Key | Built from | What it catches |
|---|---|---|
| `n:<token>` | each core-name token (≥ 2 chars) | ordinary name overlap, reordered words |
| `k:<skeleton>` | skeleton of each core token | transliterated names, vowel typos, doubled letters |
| `p:<6 chars>` | first 6 letters of the core name with spaces removed | `flint group` ↔ `flintgroup.com`, spacing differences |
| `q:<5 chars>` | first 5 letters of the skeleton of the spaceless core | the same, but tolerant to vowel noise |
| `a:<word>` | alphabetic address tokens (≥ 3 chars) | street, locality and city overlap |
| `d:<number>` | up to four address numbers (zero-stripped) | house/plot numbers regardless of position |
| `h:<num>_<word>` | house number × each of the first six address words | very selective "same number on the same street" |

Key strings are hashed to 64-bit integers, so the join works on fixed-width columns.

### 3.3 Weighted inverted-index join

Blocking runs separately for every (country, S1 → S2) and (country, S1 → S3) pair:

1. Explode records into (record, key) rows and count each key's document frequency `df` in the
   target source.
2. **Drop keys with df > 400.** A word shared by thousands of businesses (`delhi`, `services`,
   `st`) carries almost no identifying information, yet it would create hundreds of millions of
   useless pairs.
3. Give each surviving key the weight `w = ln(N_target / df)`, where N_target is the number of
   records in the target source.
4. Join S1 keys to target keys, then group by (S1 record, candidate) and compute
   **blocking score = Σ w** over shared keys, plus the number of shared keys.
5. Keep the **10 best candidates per S1 record per source**, so at most 20 per S1 entity.
6. To bound memory, S1 records are processed in chunks sized by their *expected* join
   output (Σ df over their keys), about 60 million joined rows per chunk.

A pair is proposed if it agrees on enough rare evidence of any kind: a rare surname, a house
number plus street, or a skeleton plus locality. No single field has to be perfect. That is why
this works where the one-key baseline failed.

### 3.4 Candidate pairs generated

| Split | Country | S1 records | Candidate pairs |
|---|---|---|---|
| Test | France | 259,452 | 5,168,577 |
| Test | India | 809,986 | 16,122,758 |
| Test | US | 663,106 | 13,045,112 |
| **Test total** | | **1,732,544** | **34,336,447** |
| Train | India + US | 2,206,821 | 43,354,045 |

`candidate_pairs.tsv` contains this full test candidate set, which is exactly the set the model
scores. There is one row per S1 entity with a comma-separated candidate list.

### 3.5 How we made sure true matches were not lost

We measured blocking recall on the training labels after every change: the share of true pairs
present in the candidate set. We report it per source and separately for S2/S3 records whose
name needed transliteration, so an overall average could not hide a failure on that slice.

| Blocking configuration | Recall (all) | Notes |
|---|---|---|
| Baseline: first name token + first address number, exact join | 0.448 | India ≈ 0.37, US ≈ 0.50 |
| Multi-key, df ≤ 3000, top-20 per source | 0.926 | India 0.900 (non-ASCII-name slice 0.803), US 0.943 |
| Multi-key, df ≤ 3000, top-10 per source | 0.894 | |
| **Multi-key, df ≤ 400, top-10 per source (final)** | **0.855** | same settings for train and test |

Recall@K (df ≤ 3000) shows that most of the value is in the first few candidates:

| K per source | India | India, non-ASCII names | US |
|---|---|---|---|
| 1 | 0.406 | 0.325 | 0.453 |
| 3 | 0.779 | 0.672 | 0.841 |
| 5 | 0.850 | 0.742 | 0.905 |
| 10 | 0.881 | 0.780 | 0.930 |
| 20 | 0.900 | 0.803 | 0.943 |

The final configuration (df ≤ 400, K = 10) trades about 7 points of recall for a join that is
roughly 10× faster or more. We adopted it for time reasons, and applied it identically to training and test (see
Section 5.3 for why that consistency matters). Section 6 discusses recovering the lost recall.

---

## 4. Matching Model

### 4.1 Features used (43 in total)

All string similarities come from `rapidfuzz` and are computed with `process.cpdist`, which
scores two aligned lists of strings on all CPU cores at once. On this machine that is about
34 million test pairs in roughly 10 minutes.

**Name features (18)**

| Feature | Definition |
|---|---|
| `n_ratio`, `n_tsr`, `n_wr` | ratio, token-set ratio, WRatio on the full cleaned name |
| `c_ratio`, `c_tsr`, `c_tsort`, `c_pr` | ratio, token-set, token-sort, partial ratio on the core name |
| `c_jw` | Jaro-Winkler similarity on the core name |
| `ns_ratio`, `ns_pr` | ratio and partial ratio of the core name with spaces removed (handles `flintgroup`) |
| `sk_ratio`, `sk_tsr` | ratio and token-set ratio of the skeleton strings (transliteration-robust) |
| `sk_common` | number of skeleton tokens the two names share |
| `core_seteq` | 1 if the core names contain exactly the same words in any order |
| `len1`, `len2`, `nt1`, `nt2` | core-name length in characters and in tokens, both sides |

**Address features (11)**

| Feature | Definition |
|---|---|
| `a_tsr`, `a_tsort`, `a_ratio` | token-set, token-sort and plain ratio on the cleaned address |
| `aa_tsr`, `aa_pr` | token-set and partial ratio on the alphabetic address words only |
| `num_tsr` | token-set ratio on the address numbers only |
| `hn_eq`, `hn_both`, `hn_lev` | house numbers equal / both present / Levenshtein distance |
| `a1_empty`, `a2_empty` | address missing on either side |

**Blocking and context features (14)**

| Feature | Definition |
|---|---|
| `src` | target source (2 or 3) |
| `translit` | the S2/S3 name contained non-ASCII characters |
| `is_india` | country flag (France is treated like the US) |
| `brank` | rank of the candidate among this S1's candidates from that source, by blocking score |
| `rrank` | **reverse rank**: rank of this S1 among all S1 records that proposed the same candidate |
| `bs_rel`, `bs_rrel` | blocking score divided by the best score for this S1 / for this candidate |
| `c_tsr_gap`, `a_tsr_gap` | name / address token-set score minus the best one among this S1's candidates |
| `c_tsr_rgap`, `a_tsr_rgap` | the same, but relative to the best competing S1 for this candidate |
| `frank`, `rfrank` | rank of the pair by name+address score from the S1 side / from the candidate side |
| `s1_deg` | number of candidates this S1 received |

The context features are what lift the model above a pure similarity scorer. Take two S1
businesses on the same street with similar names, both proposing the same S2 record. Their raw
similarities may look alike, but the reverse rank and reverse gaps tell the model which of the
two the record "prefers". In the final model the two strongest features by gain were `rfrank`
and `rrank`, followed by `a_tsr_rgap`, `num_tsr`, `a_tsr_gap`, `a_tsr`, `n_wr` and `sk_ratio`.

Three features that depend on the *absolute* blocking score (`bscore`, number of shared keys,
and candidate in-degree) were deliberately left out. Their scale changes with the df cut-off,
and we wanted the model to transfer across blocking settings.

### 4.2 Model

**Model type:** LightGBM gradient-boosted trees, binary objective.

| Setting | Value |
|---|---|
| Leaves / min data in leaf | 127 / 200 |
| Learning rate / rounds | 0.1 / up to 800 (early stopping, patience 50) |
| Feature fraction / bagging fraction / bagging freq | 0.8 / 0.8 / 1 |
| L2 regularization | 1.0 |
| Seed | 42 |
| Training rows | ≈ 6.5 M candidate pairs from 15 % of training S1 entities (≈ 1.0 M positives) |

We tried no class re-weighting or negative sub-sampling. With about 15 % positives among the
candidates, the imbalance is mild, and training on all candidates of the sampled entities keeps
the per-entity competition features intact.

### 4.3 Validation protocol

Leakage between pairs of the same entity is the main trap in this kind of problem, so every split
is made at the **S1-entity level**, using a fixed hash of the entity id:

- **Fit set (15 % of S1 entities):** used to train the model. Inside it, a further 10 % of
  entities (again hashed) serve as the early-stopping validation set.
- **Held-out set (10 % of S1 entities, disjoint):** never seen during fitting. Used only to
  choose the decision rule and to report the final score.
- Blocking and feature computation run over the full training corpus. Every entity, including
  the held-out ones, is therefore scored against realistic competition, as on test.

### 4.4 Threshold selection method

The macro F0.5 is computed exactly as the challenge defines it, per S1 entity: 1 if the truth and
the prediction are both empty, 0 if exactly one is empty, and set-based F0.5 otherwise. This is
averaged over **all** held-out entities, including those with no candidates and those whose true
matches were lost in blocking. On the held-out entities we then grid-search:

- the probability threshold (0.20 → 0.80 in steps of 0.10, then ±0.08 around the best in steps
  of 0.04);
- with and without the **one-to-one rule** (among pairs above the threshold, each S2/S3 id is kept
  only for the S1 with the highest probability);
- an optional lower threshold for each S1's single best candidate.

Selected rule: **threshold 0.70 with one-to-one on** (0.9170; without one-to-one 0.9170, with a
separate top-1 threshold no better). The curve is flat between 0.6 and 0.8 (0.9165–0.9170), so
the choice is not fragile. That is expected from F0.5's preference for precision: the optimum sits
where the model is already quite sure.

---

## 5. Results & Error Analysis

### 5.1 Scores

- **F_0.5 Score (macro): 0.917** on 221,351 held-out training S1 entities (≈ 4.3 M candidate
  pairs), using the exact blocking settings applied to the test set.
- Precision of submitted pairs on the held-out set: **0.991**. Pair recall: **0.832**.
- True singletons correctly left empty: **96.3 %**. Non-singletons wrongly left empty: 3.6 %.
- Public leaderboard: **0.912** for our earlier model version (Section 5.3).
- Submitted files: 1,584,360 S1 entities matched, 5,116,011 pairs; `candidate_pairs.tsv` holds the
  complete set of 34,336,447 blocked pairs scored by the model.

| Version | Candidate recall | Held-out macro F0.5 | Leaderboard |
|---|---|---|---|
| Baseline: single blocking key + weighted fuzzy score, fixed threshold 0.72 | 0.448 | 0.611 | — |
| v9: multi-key blocking (df ≤ 3000) + LightGBM, but test blocked with df ≤ 400 | 0.894 | 0.934 † | 0.912 |
| **v10 (submitted): df ≤ 400, K = 10 for both train and test** | 0.855 | **0.917** | — |

† Measured on candidates from the wider df ≤ 3000 blocking. Test had to be blocked with the
faster df ≤ 400 setting, so this number overstated what the test set would give.

**Test-set output sanity.** 1,732,544 rows. 8.6 % are predicted empty (the training singleton rate
is 5.6 %; the gap is roughly the entities whose matches were lost in blocking). Matched entities
get 3.23 ids on average (the truth is 3.67). France, which is unseen in training, behaves like the
other countries: 6.8 % empty and 3.35 matches per matched entity (India 10.1 % / 3.13,
US 7.4 % / 3.28).

### 5.2 Where the errors come from

Breaking down all true pairs of the held-out entities:

| Outcome | Share of true pairs |
|---|---|
| Found | 83.2 % |
| Never proposed by blocking | 14.5 % |
| Proposed, but scored below 0.70 | 2.3 % |
| Above threshold, but removed by the one-to-one rule | < 0.1 % |

So blocking accounts for about six out of every seven missed pairs. The classifier itself is
already precise and rarely misses a pair it is shown.

**Common false positives (wrong merges).** They are rare (0.9 % of predicted pairs) and mostly
fall into three patterns:
- *Near-identical names at the same address that are genuinely different entities*, e.g.
  `Yay Breeze Inc` vs. `YY Breeze Inc` at the same street number, or `Reid Kayley DO` vs.
  `Reid Kayley SO`. The generator swaps a single letter to create a distinct business, and
  our similarity features cannot tell that apart from a typo.
- *Same address, overlapping words, different business*: `Maninagar Leasing` vs.
  `Maninagar Software`, or `Federal Tree LP` vs. `Federal Center` on the same street.
- *Sibling entities from a family of names*: `Narayan Engineering` vs.
  `Narayan Engineering Industries` with a house number differing in the last digit.

**Common false negatives (missed matches).**
- *Not proposed by blocking* (most of the loss). Typical cases are an **empty S2/S3 address combined
  with a corrupted name** (`Frontier Newbridge` → `frontier nwbrmege`, no address), a
  **transliterated name whose address is written differently** (`Digital Finance Private Limited`
  → `ddijittl phaainens praaivett limittedd`), and a **fully concatenated name** whose prefix
  differs (`trinitylutheranchurch`). Some pairs are easy but lost the competition for the 10
  slots: `Better Ingredients` vs. `Better Ingredients` with house numbers 547 vs. 54 loses to
  other candidates, because both name words are common enough to be cut by df ≤ 400.
- *Proposed but scored low*: the **house number changed** and a **word was added or removed**
  (`Koppen Associates of Eagan, 2272 Whispering Trl` vs. `Koppen Associates Eagan,
  2389 Whispering Trl`), or the **name was replaced by an unrelated token** while the address still
  matches (`Systems Aquarius Mill Private Limited` → `Rizayuma`).

### 5.3 A lesson about train/test consistency

Our first submitted model (v9) was trained on candidates from the wider blocking setting
(df ≤ 3000), but the test set was blocked with df ≤ 400 to meet the deadline. Its held-out
estimate (0.934) came out well above its leaderboard score (0.912). We re-ran training blocking
with the test settings and scored both models on the same held-out entities: v9 0.9164, v10
0.9170. The model itself transfers well. Almost all of the gap comes from the **smaller candidate
set** (recall 0.894 → 0.855). Two practical conclusions: estimate your score with the exact
blocking used at inference, and blocking recall is the lever to pull next.

---

## 6. Conclusion

Most of our progress came from candidate generation. Replacing a single exact blocking key with
weighted multi-key blocking over normalized tokens and consonant skeletons roughly doubled the
share of true pairs that reach the matcher. A gradient-boosted classifier that sees each pair in
the context of its competitors then turned that recall into a macro F0.5 of about 0.917 at 99 %
precision. The main lesson is to measure blocking recall first and keep every setting identical
between training and inference. With more time, the clearest next step is restoring the wider
blocking (df ≤ 3000, K = 20: recall 0.926 instead of 0.855). Error analysis suggests that alone
should recover most of the remaining gap.

---

## Appendix

### A. Code Artefacts

The complete pipeline ships under `code/business_entity_resolution/`:

```
business_entity_resolution/
├── README.md              # how to reproduce end to end (data → blocking → matching → output)
├── requirements.txt       # pinned library versions
├── models/
│   ├── lgb_model.txt      # the trained LightGBM model used for the submission
│   └── best_params.json   # chosen decision rule (threshold 0.70, one-to-one)
└── src/
    ├── config.py          # paths (ER_DATA / ER_WORK / ER_OUT) and pipeline settings
    ├── run_all.py         # entry point: runs every stage in order
    ├── prep.py            # Stage 1: read TSVs, normalize, transliterate, skeletons, blocking keys
    ├── block.py           # Stage 2: IDF-weighted inverted-index blocking, top-K per S1 per source
    ├── eval_block.py      # blocking recall on the training labels (overall and non-ASCII slice)
    ├── features.py        # Stage 3: pairwise similarity features (rapidfuzz, multithreaded)
    ├── train.py           # Stage 4: group features, LightGBM, held-out threshold tuning
    └── predict.py         # Stage 5: score test candidates, decision layer, write output/*.tsv
```

**Entry points.**
- `python run_all.py` (from `src/`, with the six raw TSVs and `train_ground_truth.tsv` in
  `ER_DATA`) reproduces everything: prep → blocking (train, test) → blocking recall report →
  features → training → prediction. It writes `output/matching_results.tsv` and
  `output/candidate_pairs.tsv`.
- `python run_all.py --predict` skips training and uses the shipped model in `models/`. It only
  needs the three test TSVs.

The submitted `output/` files were produced by exactly this packaged code: `run_all.py --predict`
run from the raw test TSVs in a clean working folder (prep → blocking → features → prediction,
about 50 minutes). Compared with our earlier leaderboard run, which read the same records in a
different row order, 94 % of S1 rows are identical and the totals agree (5,116,011 vs. 5,116,131
predicted pairs). The remaining differences come from ties in the blocking score: when equal-scoring
candidates straddle the top-10 cut, input order decides which ones are kept.

**Runtime** (12-core laptop, 16 GB RAM; peak memory up to about 8 GB, in blocking):

| Stage | Train | Test |
|---|---|---|
| prep | ≈ 6 min | ≈ 5 min |
| block | ≈ 5 min | ≈ 5 min |
| features | ≈ 11 min | ≈ 10 min |
| train (fit + tuning) | ≈ 6 min | — |
| predict + write | — | ≈ 7 min |

### B. Additional Results

**Decision threshold sweep on the held-out set (one-to-one on)**

| Threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| Macro F0.5 | 0.9041 | 0.9096 | 0.9130 | 0.9151 | 0.9165 | **0.9170** | 0.9164 |

**LightGBM training curve (early-stopping validation logloss):** 0.0156 at 100 rounds,
0.0142 at 200, 0.0134 at 400, 0.0130 at 800 (still improving slightly; capped for time).

**Things we tried or considered and did not use.**
- *Translation of native-script names:* unnecessary, because the names are phonetic spellings.
  Rule-based transliteration plus skeletons is cheaper and more faithful to proper nouns.
- *A fixed hand-weighted fuzzy score with one threshold* (the baseline): easy to tune, but it
  cannot combine many weak signals, and it has no notion of competing candidates.
- *Dense sentence embeddings for blocking:* heavy for 12 million records on a laptop, and the
  lexical keys already reached 0.93 recall in their wider configuration.

**Known limitations and next steps.**
1. Blocking recall: revert to df ≤ 3000 and K = 20 (0.926 vs. 0.855), or add a second-pass
   blocker only for S1 entities whose best candidate score is low.
2. Legal forms written with separators (`L.L.C.` → `l l c`) are not merged back into `llc`.
3. The single-letter `r → rue` mapping, meant for French addresses, also fires on Indian
   initials such as `K R Road`. It is harmless for matching (both sides are mapped the same way),
   but it is untidy.
4. Pair-level decisions ignore transitivity among S2/S3 duplicates. A clustering step could add
   missing duplicates of an already-matched record.

**External data / APIs:** none. All processing is local and uses only open-source libraries
(polars, pyarrow, numpy, rapidfuzz, Unidecode, LightGBM). No pretrained models are used and no
network calls are made. Randomness is seeded (42).
