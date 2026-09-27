"""
v9/features.py — Pairwise features for blocked candidates (chunked, multithreaded).

Output: WORK/feat_<split>_<country>/part_XXX.parquet
"""
import gc, os, sys, time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

WORK = Path(os.environ["ER_WORK"])
CHUNK = 3_000_000

STR_COLS = ["name", "core", "skel", "addr", "hn", "addr_alpha", "addr_nums"]


def cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def pair_features(d):
    """d has <col>_1 and <col>_2 string columns for STR_COLS."""
    g = {c: [d[f"{c}_1"].to_list(), d[f"{c}_2"].to_list()] for c in STR_COLS}
    nsp1 = [s.replace(" ", "") for s in g["core"][0]]
    nsp2 = [s.replace(" ", "") for s in g["core"][1]]
    f = {
        "n_ratio": cp(fuzz.ratio, *g["name"]),
        "n_tsr": cp(fuzz.token_set_ratio, *g["name"]),
        "n_wr": cp(fuzz.WRatio, *g["name"]),
        "c_ratio": cp(fuzz.ratio, *g["core"]),
        "c_tsr": cp(fuzz.token_set_ratio, *g["core"]),
        "c_tsort": cp(fuzz.token_sort_ratio, *g["core"]),
        "c_pr": cp(fuzz.partial_ratio, *g["core"]),
        "c_jw": cp(JaroWinkler.normalized_similarity, *g["core"]),
        "ns_ratio": cp(fuzz.ratio, nsp1, nsp2),
        "ns_pr": cp(fuzz.partial_ratio, nsp1, nsp2),
        "sk_ratio": cp(fuzz.ratio, *g["skel"]),
        "sk_tsr": cp(fuzz.token_set_ratio, *g["skel"]),
        "a_tsr": cp(fuzz.token_set_ratio, *g["addr"]),
        "a_tsort": cp(fuzz.token_sort_ratio, *g["addr"]),
        "a_ratio": cp(fuzz.ratio, *g["addr"]),
        "aa_tsr": cp(fuzz.token_set_ratio, *g["addr_alpha"]),
        "aa_pr": cp(fuzz.partial_ratio, *g["addr_alpha"]),
        "num_tsr": cp(fuzz.token_set_ratio, *g["addr_nums"]),
        "hn_lev": cp(Levenshtein.distance, *g["hn"]),
    }
    out = pl.DataFrame(f)
    out = out.with_columns(
        (d["hn_1"] == d["hn_2"]).cast(pl.Int8).alias("hn_eq"),
        ((d["hn_1"] != "") & (d["hn_2"] != "")).cast(pl.Int8).alias("hn_both"),
        (d["addr_2"] == "").cast(pl.Int8).alias("a2_empty"),
        (d["addr_1"] == "").cast(pl.Int8).alias("a1_empty"),
        d["core_1"].str.len_chars().cast(pl.Int16).alias("len1"),
        d["core_2"].str.len_chars().cast(pl.Int16).alias("len2"),
        d["core_1"].str.count_matches(" ").cast(pl.Int8).alias("nt1"),
        d["core_2"].str.count_matches(" ").cast(pl.Int8).alias("nt2"),
        # exact core-token set equality
        (d["core_1"].str.split(" ").list.sort() == d["core_2"].str.split(" ").list.sort())
            .cast(pl.Int8).alias("core_seteq"),
        (d["skel_1"].str.split(" ").list.set_intersection(d["skel_2"].str.split(" ")).list.len()
            .cast(pl.Int8).alias("sk_common")),
    )
    return out


def load_side(path, cols, suffix):
    return pl.read_parquet(path, columns=["idx"] + cols).rename(
        {c: f"{c}{suffix}" for c in ["idx"] + cols})


def main():
    split = sys.argv[1]
    for cpath in sorted(WORK.glob(f"cands_{split}_*.parquet")):
        country = cpath.stem.split("_", 2)[2]
        odir = WORK / f"feat_{split}_{country}"
        if (odir / "_DONE").exists():
            continue
        odir.mkdir(exist_ok=True)
        t0 = time.time()
        c = pl.read_parquet(cpath)
        # group-level blocking features
        c = c.with_columns(
            pl.col("bscore").rank("ordinal", descending=True).over("cand", "src").cast(pl.Int16).alias("rrank"),
            pl.len().over("cand", "src").cast(pl.Int16).alias("cand_deg"),
            (pl.col("bscore") / pl.col("bscore").max().over("s1", "src")).alias("bs_rel"),
            (pl.col("bscore") / pl.col("bscore").max().over("cand", "src")).alias("bs_rrel"),
        ).sort("s1", "src", "brank")
        s1 = load_side(WORK / f"{split}_s1_{country}.parquet", STR_COLS, "_1")
        sx = {k: load_side(WORK / f"{split}_s{k}_{country}.parquet", STR_COLS + ["translit"], "_2")
              for k in (2, 3)}
        n = c.height
        print(f"[{split} {country}] {n:,} pairs", flush=True)
        for i, st in enumerate(range(0, n, CHUNK)):
            op = odir / f"part_{i:03d}.parquet"
            if op.exists():
                continue
            ch = c.slice(st, CHUNK)
            d = ch.join(s1, left_on="s1", right_on="idx_1", how="left")
            parts = []
            for k in (2, 3):
                parts.append(d.filter(pl.col("src") == k)
                              .join(sx[k], left_on="cand", right_on="idx_2", how="left"))
            d = pl.concat(parts).with_columns([pl.col(f"{x}_{s}").fill_null("") for x in STR_COLS for s in (1, 2)])
            feats = pair_features(d)
            base = d.select("s1", "cand", "src", "bscore", "nk", "brank", "rrank", "cand_deg",
                            "bs_rel", "bs_rrel", pl.col("translit_2").cast(pl.Int8).alias("translit"))
            pl.concat([base, feats], how="horizontal").write_parquet(op)
            print(f"  part {i} ({time.time()-t0:.0f}s)", flush=True)
            del d, feats, base, parts
            gc.collect()
        (odir / "_DONE").touch()
        del c, s1, sx
        gc.collect()


if __name__ == "__main__":
    main()
