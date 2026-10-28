"""
train.py — Stage 4: LightGBM on blocked-candidate features; tune decision layer on held-out S1.

Split S1 entities by hash: FIT_FRAC go to model fitting, the rest are a held-out set on
which we predict, tune the threshold and report macro F0.5 (honest estimate).
"""
import gc, json, os, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK, DATA, FIT_FRAC, HO_FRAC, SEED, K as KMAX

# bscore/nk/cand_deg depend on blocking MAXDF (test uses a faster setting) -> excluded
DROP = {"s1", "cand", "y", "s1_id", "cand_id", "fit", "bscore", "nk", "cand_deg"}
COVERED = []


GKEYS = ["s1", "cand", "src"]


def group_feats(fd):
    """Cross-row features over the FULL candidate set of a country (narrow columns only)."""
    g = (pl.scan_parquet(fd / "part_*.parquet").filter(pl.col("brank") <= KMAX)
           .select(*GKEYS, "c_tsr", "a_tsr").collect())
    sc = pl.col("c_tsr") + pl.col("a_tsr")
    return g.with_columns(
        (pl.col("c_tsr") - pl.col("c_tsr").max().over("s1")).alias("c_tsr_gap"),
        (pl.col("a_tsr") - pl.col("a_tsr").max().over("s1")).alias("a_tsr_gap"),
        (pl.col("c_tsr") - pl.col("c_tsr").max().over("cand", "src")).alias("c_tsr_rgap"),
        (pl.col("a_tsr") - pl.col("a_tsr").max().over("cand", "src")).alias("a_tsr_rgap"),
        sc.rank("ordinal", descending=True).over("s1", "src").cast(pl.Int16).alias("frank"),
        sc.rank("ordinal", descending=True).over("cand", "src").cast(pl.Int16).alias("rfrank"),
        pl.len().over("s1").cast(pl.Int16).alias("s1_deg"),
    ).drop("c_tsr", "a_tsr")


def load_split(split, country, s1_ids=None):
    """Load features for one country; optionally only for S1 entity ids in s1_ids."""
    fd = WORK / f"feat_{split}_{country}"
    ids1 = pl.read_parquet(WORK / f"{split}_s1_{country}.parquet", columns=["idx", "entity_id"])              .rename({"idx": "s1", "entity_id": "s1_id"})
    # only S1s whose features were fully computed (parts are written in s1 order)
    smax = pl.scan_parquet(fd / "part_*.parquet").select(pl.col("s1").max()).collect().item()
    if not (fd / "_DONE").exists():
        ids1 = ids1.filter(pl.col("s1") < smax)
    COVERED.append(ids1.select("s1_id"))
    if s1_ids is not None:
        ids1 = ids1.join(s1_ids, on="s1_id", how="semi")
    gf = group_feats(fd).join(ids1.select("s1"), on="s1", how="semi")
    df = (pl.scan_parquet(fd / "part_*.parquet").filter(pl.col("brank") <= KMAX)
            .join(ids1.lazy(), on="s1", how="inner")
            .collect().join(gf, on=GKEYS, how="left"))
    del gf
    idsx = pl.concat([pl.read_parquet(WORK / f"{split}_s{k}_{country}.parquet", columns=["idx", "entity_id"])
                      .with_columns(pl.lit(k, pl.Int8).alias("src")) for k in (2, 3)])
    df = df.join(idsx.rename({"idx": "cand", "entity_id": "cand_id"}), on=["cand", "src"], how="left")
    return df.with_columns(pl.lit(1 if country == "India" else 0, pl.Int8).alias("is_india"))


def truth_table():
    gt = pl.read_csv(DATA / "train_ground_truth.tsv", separator="	", quote_char=None,
                     infer_schema=False).rename({"source1_entity_id": "s1_id"})
    t = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("c"))
           .explode("c").filter(pl.col("c").str.len_chars() > 0)
           .select("s1_id", pl.col("c").alias("cand_id")))
    return gt.select("s1_id"), t


def bucket(col):
    return pl.col(col).hash(seed=7) % 1000


def is_fit(col):
    return bucket(col) < int(FIT_FRAC * 1000)


def is_ho(col):  # held-out evaluation S1s (disjoint from fit)
    return bucket(col) >= 1000 - int(HO_FRAC * 1000)


def macro_f05(pred, s1_all, tsize):
    """pred: s1_id,cand_id,y (y=1 if true). s1_all: all s1_ids evaluated; tsize: s1_id,nt."""
    g = pred.group_by("s1_id").agg(pl.len().alias("np"), pl.col("y").sum().alias("tp"))
    d = (s1_all.join(tsize, on="s1_id", how="left").join(g, on="s1_id", how="left")
         .with_columns(pl.col("nt").fill_null(0), pl.col("np").fill_null(0), pl.col("tp").fill_null(0)))
    p = pl.col("tp") / pl.max_horizontal(pl.col("np"), 1)
    r = pl.col("tp") / pl.max_horizontal(pl.col("nt"), 1)
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0) \
          .when((pl.col("nt") == 0) | (pl.col("np") == 0)).then(0.0) \
          .otherwise((1.25 * p * r / (0.25 * p + r)).fill_nan(0.0))
    return d.select(f.mean()).item()


def decide(df, thr, one_to_one=True, cap=None, top1_thr=None):
    x = df.filter(pl.col("p") >= min(thr, top1_thr or thr))
    if one_to_one:
        x = x.filter(pl.col("p") == pl.col("p").max().over("cand_id"))
    if top1_thr is not None:  # the best candidate of an S1 passes at top1_thr, the rest need thr
        best = pl.col("p") == pl.col("p").max().over("s1_id")
        x = x.filter((best & (pl.col("p") >= top1_thr)) | (pl.col("p") >= thr))
    if cap:
        x = x.filter(pl.col("p").rank("ordinal", descending=True).over("s1_id") <= cap)
    return x


def main():
    t0 = time.time()
    s1_all, truth = truth_table()
    truth = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
    tsize = truth.group_by("s1_id").len("nt")
    frames = []
    sub = s1_all.filter(is_fit("s1_id") | is_ho("s1_id"))
    for country in ("India", "US"):
        df = load_split("train", country, sub)
        df = df.join(truth, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))
        frames.append(df)
        print(f"{country}: {df.height:,} pairs, pos {df['y'].sum():,} ({time.time()-t0:.0f}s)", flush=True)
    df = pl.concat(frames, how="diagonal_relaxed")
    del frames
    gc.collect()
    covered = pl.concat(COVERED)
    rec = df["y"].sum() / truth.join(sub, on="s1_id", how="semi").join(covered, on="s1_id", how="semi").height
    print(f"candidate recall: {rec:.4f}", flush=True)

    df = df.with_columns(is_fit("s1_id").alias("fit"))
    feats = [c for c in df.columns if c not in DROP]
    print("features:", feats)
    fit = df.filter(pl.col("fit"))
    X = fit.select(feats).to_numpy().astype(np.float32)
    y = fit["y"].to_numpy()
    # inner early-stopping split by s1 hash
    inner = (fit.select((pl.col("s1_id").hash(seed=11) % 10) == 0).to_series().to_numpy())
    del fit
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=SEED, verbose=-1, num_threads=os.cpu_count())
    dtr = lgb.Dataset(X[~inner], y[~inner], feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X[inner], y[inner], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    del X, y, dtr, dva
    gc.collect()
    m.save_model(str(WORK / "lgb_model.txt"))
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", [(f, round(g / 1e3)) for f, g in imp[:15]])

    ho = df.filter(~pl.col("fit")).select("s1_id", "cand_id", "y", *feats)
    del df
    gc.collect()
    p = np.concatenate([m.predict(ho.slice(i, 4_000_000).select(feats).to_numpy().astype(np.float32),
                                  num_threads=os.cpu_count()) for i in range(0, ho.height, 4_000_000)])
    ho = ho.select("s1_id", "cand_id", "y").with_columns(pl.Series("p", p))
    ho.write_parquet(WORK / "holdout_preds.parquet")
    s1_ho = s1_all.filter(is_ho("s1_id")).join(covered, on="s1_id", how="semi")
    print(f"holdout S1: {s1_ho.height:,}  pairs {ho.height:,}  ({time.time()-t0:.0f}s)", flush=True)

    best = (0, None)
    for oto in (True, False):
        for thr in np.arange(0.20, 0.81, 0.10):
            s = macro_f05(decide(ho, thr, oto), s1_ho, tsize)
            print(f"  oto={oto} thr={thr:.2f}  F0.5={s:.5f}", flush=True)
            if s > best[0]:
                best = (s, dict(thr=float(thr), one_to_one=oto, top1_thr=None))
    b = best[1]
    for thr in np.arange(b["thr"] - 0.08, b["thr"] + 0.081, 0.04):
        for t1 in (None, 0.2):
            s = macro_f05(decide(ho, thr, b["one_to_one"], top1_thr=t1), s1_ho, tsize)
            if s > best[0]:
                best = (s, dict(thr=float(thr), one_to_one=b["one_to_one"], top1_thr=t1))
    print(f"BEST holdout macro F0.5 = {best[0]:.5f}  {best[1]}", flush=True)
    json.dump({"f05": best[0], **best[1], "cand_recall": rec}, open(WORK / "best_params.json", "w"))


if __name__ == "__main__":
    main()
