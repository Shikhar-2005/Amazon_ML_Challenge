"""
v9/block.py — IDF-weighted multi-key blocking.

For each (split, country, source∈{2,3}): every record contributes a set of keys
(name tokens, consonant skeletons, spaceless-name prefixes, address tokens, numbers,
house-number+street composites). An S1/SX pair's blocking score is the sum of
log(N/df) over the keys they share. Keep top-K per S1 per source.

Output: WORK/cands_<split>_<country>.parquet
  s1, cand (int32 row idx into prep parquet), src (2/3), bscore, nk, brank
"""
import gc, math, os, sys, time
from pathlib import Path

import polars as pl

WORK = Path(os.environ["ER_WORK"])
K = int(os.environ.get("ER_K", 20))
MAXDF = int(os.environ.get("ER_MAXDF", 3000))
PAIRS_PER_CHUNK = 60_000_000


def key_table(path):
    return (pl.scan_parquet(path).select("idx", "keys").explode("keys")
            .filter(pl.col("keys").is_not_null())
            .select("idx", pl.col("keys").hash(seed=42).alias("h")).collect())


def block_pair(s1k, sxk, n_sx):
    dfx = sxk.group_by("h").len("dfx").filter(pl.col("dfx") <= MAXDF)
    df1 = s1k.group_by("h").len("df1")
    kk = dfx.join(df1, on="h").with_columns(
        (pl.lit(n_sx, pl.Float64) / pl.col("dfx")).log().cast(pl.Float32).alias("w"),
        (pl.col("dfx") * pl.col("df1")).alias("cost"))
    total = kk["cost"].sum()
    print(f"    keys kept {kk.height:,}  join rows {total:,}", flush=True)
    sx_small = sxk.join(kk.select("h", "w"), on="h")
    s1_small = s1k.join(kk.select("h"), on="h", how="semi")
    # chunk S1 by idx so each join stays bounded
    cost_per_s1 = s1_small.join(kk.select("h", "cost"), on="h").group_by("idx").agg(pl.col("cost").sum())
    cost_per_s1 = cost_per_s1.sort("idx").with_columns(
        (pl.col("cost").cum_sum() // PAIRS_PER_CHUNK).alias("chunk"))
    chunk_of = cost_per_s1.select("idx", "chunk")
    s1_small = s1_small.join(chunk_of, on="idx")
    out = []
    for c in range(int(chunk_of["chunk"].max()) + 1 if chunk_of.height else 0):
        q = s1_small.filter(pl.col("chunk") == c).select("idx", "h")
        j = (q.join(sx_small, on="h", suffix="_x")
               .group_by("idx", "idx_x").agg(pl.col("w").sum().alias("bscore"), pl.len().alias("nk"))
               .with_columns(pl.col("bscore").rank("ordinal", descending=True).over("idx")
                             .cast(pl.Int16).alias("brank"))
               .filter(pl.col("brank") <= K))
        out.append(j)
        del j
    gc.collect()
    res = pl.concat(out) if out else pl.DataFrame()
    return res.rename({"idx": "s1", "idx_x": "cand"}).with_columns(pl.col("nk").cast(pl.Int16))


def main():
    split = sys.argv[1]
    countries = sorted({p.stem.split("_", 2)[2] for p in WORK.glob(f"{split}_s1_*.parquet")})
    for country in countries:
        out_path = WORK / f"cands_{split}_{country}.parquet"
        if out_path.exists():
            continue
        t0 = time.time()
        print(f"[{split} {country}]", flush=True)
        s1k = key_table(WORK / f"{split}_s1_{country}.parquet")
        parts = []
        for src in (2, 3):
            p = WORK / f"{split}_s{src}_{country}.parquet"
            n_sx = pl.scan_parquet(p).select(pl.len()).collect().item()
            sxk = key_table(p)
            r = block_pair(s1k, sxk, n_sx).with_columns(pl.lit(src, pl.Int8).alias("src"))
            print(f"    S{src}: {r.height:,} cands ({time.time()-t0:.0f}s)", flush=True)
            parts.append(r)
            del sxk
            gc.collect()
        pl.concat(parts).write_parquet(out_path)
        del s1k, parts
        gc.collect()


if __name__ == "__main__":
    main()
