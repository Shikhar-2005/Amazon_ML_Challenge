# Business Entity Resolution Pipeline

An end-to-end Machine Learning pipeline for large-scale record linkage and duplicate business entity resolution. The system matches entity records across multiple disparate data sources (Source 1 to Source 2 and Source 3) using multi-key blocking, pairwise similarity features, and a tuned LightGBM classifier with one-to-one constraint optimization.

The complete methodology, algorithmic choices, and evaluation metrics are documented in [Documentation_template.md](Documentation_template.md).

---

## 1. System Requirements & Environment

- **Python:** 3.10 or newer (tested on Python 3.11 / 3.14 on Windows and Linux)
- **Hardware:** ~16 GB RAM and ~20 GB free disk space for intermediate Parquet files.
- **Compute:** CPU-only (multithreaded using Polars, NumPy, and LightGBM). No GPU or internet connection required.

```bash
pip install -r requirements.txt
```

---

## 2. Data Layout

Place the source datasets in a folder (default: `./data`) or configure the path via the `ER_DATA` environment variable:

```text
data/
├── train_source1.tsv   train_source2.tsv   train_source3.tsv
├── train_ground_truth.tsv
└── test_source1.tsv    test_source2.tsv    test_source3.tsv
```

### Environment Variables & Settings

All paths and hyperparameters can be overridden via environment variables (defined in `src/config.py`):

| Variable | Description | Default |
|---|---|---|
| `ER_DATA` | Folder containing raw input TSVs | `./data` |
| `ER_WORK` | Scratch folder for intermediate Parquet files & cache | `./work` |
| `ER_OUT` | Directory where output TSVs are generated | `./output` |
| `ER_MAXDF` | Max document frequency threshold for blocking keys | `400` |
| `ER_K` | Number of candidate pairs retained per entity per source | `10` |
| `ER_FIT_FRAC` | Proportion of entities used for model training | `0.15` |
| `ER_HO_FRAC` | Proportion of entities used for held-out validation / tuning | `0.10` |

---

## 3. Pipeline Execution

Navigate to the `src/` directory:

```bash
cd src
```

### Option A: Full Pipeline (Data Prep -> Blocking -> Features -> Training -> Prediction)

```bash
# Optional: point to custom data folder
# Windows PowerShell: $env:ER_DATA="C:\path\to\data"
# Linux/macOS:        export ER_DATA="/path/to/data"

python run_all.py
```

`run_all.py` executes each stage sequentially and caches intermediate results to `./work/`:

| Stage | Script | Purpose |
|---|---|---|
| 1 | `prep.py` | Text normalization, address cleaning, unidecode transliteration, and blocking key generation. |
| 2 | `block.py` | Inverted-index candidate generation with IDF weighting (top-K per source). |
| 3 | `eval_block.py` | Evaluates candidate recall against ground truth labels (including transliterated records). |
| 4 | `features.py` | Extracts 40+ pairwise text, token, and rank competition features. |
| 5 | `train.py` | Trains LightGBM model and tunes macro F0.5 decision thresholds on validation splits. |
| 6 | `predict.py` | Scores test candidate pairs, applies 1-to-1 decision rules, and exports final TSV files. |

### Option B: Fast Inference Only (Using Pretrained Model)

To run inference directly on the test set using the pretrained model weights in `models/`:

```bash
python run_all.py --predict
```

---

## 4. Output Files

Inference results are saved to the `output/` directory:
- `matching_results.tsv`: `source1_entity_id <TAB> matched_entity_ids` (comma-separated list of identified duplicates).
- `candidate_pairs.tsv`: `source1_entity_id <TAB> candidate_entity_ids` (all candidate pairs considered by the model).

---

## 5. Repository Structure

```text
├── .gitignore
├── requirements.txt         # Project dependencies
├── README.md                # Project overview and run guide
├── Documentation_template.md# In-depth architectural & algorithmic documentation
├── models/
│   ├── best_params.json     # Tuned decision thresholds & metrics
│   └── lgb_model.txt        # Pretrained LightGBM model
└── src/
    ├── config.py            # Global paths, parameters, and directories
    ├── prep.py              # Data preprocessing & transliteration
    ├── block.py             # Inverted-index candidate blocking
    ├── eval_block.py        # Recall@K evaluation
    ├── features.py          # Similarity & competition feature extraction
    ├── train.py             # LightGBM training & threshold optimizer
    ├── predict.py           # Model scoring & TSV generation
    └── run_all.py           # Pipeline runner
```