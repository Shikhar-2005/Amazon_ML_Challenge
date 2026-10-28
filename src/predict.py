"""
predict.py — Stage 5: score test candidates with the trained LightGBM model, apply the tuned
decision layer (threshold + one-to-one), write matching_results.tsv / candidate_pairs.tsv.
"""
import gc, json, os, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT  # noqa: E402
from train import WORK, load_split, decide  # noqa: E402

CAND_MIN_P = 0.0  # candidate_pairs.tsv = the full blocked candidate set that was fed to the model


def main():
    t0 = time.time()
    # Use a freshly trained model if train.py was run, else the shipped one in ../models/
    mdir = WORK if (WORK / "lgb_model.txt").exists() else Path(__file__).resolve().parent.parent / "models"
    m = lgb.Booster(model_file=str(mdir / "lgb_model.txt"))
    feats = m.feature_name()
    bp = json.load(open(mdir / "best_params.json"))
    print("model from:", mdir, flush=True)
    print("params:", bp, flush=True)
    scored, all_s1 = [], []
    countries = sorted({p.stem.split("_", 2)[2] for p in WORK.glob("test_s1_*.parquet")})
    for country in countries:
        all_s1.append(pl.read_parquet(WORK / f"test_s1_{country}.parquet", columns=["entity_id"]))
        df = load_split("test", country)
        p = np.concatenate([m.predict(df.slice(i, 3_000_000).select(feats).to_numpy().astype(np.float32),
                                      num_threads=os.cpu_count()) for i in range(0, df.height, 3_000_000)])
        scored.append(df.select("s1_id", "cand_id").with_columns(pl.Series("p", p.astype(np.float32))))
        print(f"{country}: {df.height:,} pairs scored ({time.time()-t0:.0f}s)", flush=True)
        del df
        gc.collect()
    sc = pl.concat(scored).filter(pl.col("p") >= CAND_MIN_P)
    all_s1 = pl.concat(all_s1).rename({"entity_id": "s1_id"}).unique()

    pred = decide(sc, bp["thr"], bp["one_to_one"], top1_thr=bp.get("top1_thr"))

    def write(frame, header, path):
        g = frame.group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias("ids"))
        out = all_s1.join(g, on="s1_id", how="left").with_columns(pl.col("ids").fill_null("")).sort("s1_id")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(header + "\n")
            for a, b in out.iter_rows():
                f.write(f"{a}\t{b}\n")

    write(pred, "source1_entity_id\tmatched_entity_ids", OUT / "matching_results.tsv")
    write(sc, "source1_entity_id\tcandidate_entity_ids", OUT / "candidate_pairs.tsv")
    n = all_s1.height
    nm = pred["s1_id"].n_unique()
    print(f"S1 {n:,}  matched {nm:,} ({100*nm/n:.1f}%)  pairs {pred.height:,}  "
          f"avg/matched {pred.height/max(nm,1):.2f}  empty {100*(n-nm)/n:.1f}%  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
