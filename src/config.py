"""
Shared paths and pipeline settings. Everything can be overridden with environment variables.

ER_DATA : folder holding the raw TSVs (train_source{1,2,3}.tsv, train_ground_truth.tsv,
          test_source{1,2,3}.tsv). Default: the folder three levels above this file,
          i.e. the submission root's parent layout  <root>/code/business_entity_resolution/src.
ER_WORK : scratch folder for intermediate Parquet files, candidates, features and the model.
ER_OUT  : where matching_results.tsv and candidate_pairs.tsv are written.
"""
import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent          # business_entity_resolution/
_ROOT = _PKG.parent.parent                              # submission root

DATA = Path(os.environ.get("ER_DATA", _ROOT / "data"))
WORK = Path(os.environ.get("ER_WORK", _PKG / "work"))
OUT = Path(os.environ.get("ER_OUT", _ROOT / "output"))

# Blocking: keys shared by more than MAXDF records of the target source are ignored;
# the K best candidates per S1 record per source are kept. Train and test MUST use the same values.
MAXDF = int(os.environ.get("ER_MAXDF", 400))
K = int(os.environ.get("ER_K", 10))

# Share of training S1 entities used to fit the model / to tune the decision threshold
# (disjoint, chosen by a fixed hash of the entity id).
FIT_FRAC = float(os.environ.get("ER_FIT_FRAC", 0.15))
HO_FRAC = float(os.environ.get("ER_HO_FRAC", 0.10))

SEED = 42

for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)
