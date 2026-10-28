"""eval_block.py — blocking recall@K on train, overall and for transliterated candidates."""
import os
from pathlib import Path

import polars as pl

from config import WORK, DATA

gt = pl.read_csv(DATA / "train_ground_truth.tsv", separator="	", quote_char=None,
                 infer_schema=False).rename({"source1_entity_id": "s1_id"})
truth = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("c"))
           .explode("c").filter(pl.col("c").str.len_chars() > 0)
           .select("s1_id", pl.col("c").alias("cand_id"), pl.lit(1, pl.Int8).alias("y")))
tot = hit = 0
for country in ("India", "US"):
    c = pl.read_parquet(WORK / f"cands_train_{country}.parquet")
    ids1 = pl.read_parquet(WORK / f"train_s1_{country}.parquet", columns=["idx", "entity_id"])
    idsx = pl.concat([pl.read_parquet(WORK / f"train_s{k}_{country}.parquet", columns=["idx", "entity_id", "translit"])
                      .with_columns(pl.lit(k, pl.Int8).alias("src")) for k in (2, 3)])
    c = (c.join(ids1.rename({"idx": "s1", "entity_id": "s1_id"}), on="s1")
          .join(idsx.rename({"idx": "cand", "entity_id": "cand_id"}), on=["cand", "src"]))
    t = truth.join(ids1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi") \
             .join(idsx.select(pl.col("entity_id").alias("cand_id"), "translit"), on="cand_id")
    m = t.join(c.select("s1_id", "cand_id", "brank"), on=["s1_id", "cand_id"], how="left")
    print(f"[{country}] true pairs {t.height:,}  cands {c.height:,} ({c.height / ids1.height:.1f}/S1)")
    for k in (1, 3, 5, 10):
        r = (m["brank"] <= k).sum() / t.height
        rt = m.filter(pl.col("translit"))
        rtr = (rt["brank"] <= k).sum() / max(rt.height, 1)
        print(f"   recall@{k:<2} per source: {r:.4f}   translit: {rtr:.4f}")
    tot += t.height
    hit += m["brank"].is_not_null().sum()
print(f"OVERALL blocking recall: {hit / tot:.4f}")
