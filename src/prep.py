"""
prep.py — Stage 1: read the raw TSVs, normalize names/addresses, transliterate non-Latin
script, build consonant skeletons and blocking keys.

Input : DATA/{train,test}_source{1,2,3}.tsv
Output: WORK/<split>_s<k>_<country>.parquet   (one file per split, source and country)
"""
import os, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from unidecode import unidecode

from config import DATA, WORK

# ─── vocab ──────────────────────────────────────────────────────────────────────
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

LEGAL = {"pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
         "co", "company", "llp", "lp", "plc", "pc", "the", "and", "of", "sarl", "sas", "sa", "eurl",
         "sci", "gmbh", "dba", "m", "s", "ms", "mr", "mrs", "sri", "shri", "shree", "com", "www",
         "in", "net", "org", "pllc", "pa", "ltda", "cie", "et", "de", "du", "des", "la", "le", "les"}

STATES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    # India
    "maharashtra": "mh", "karnataka": "ka", "haryana": "hr", "delhi": "dl", "bihar": "br",
    "gujarat": "gj", "rajasthan": "rj", "punjab": "pb", "kerala": "kl", "keralam": "kl",
    "telangana": "tg", "ts": "tg", "odisha": "od", "orissa": "od", "assam": "as", "goa": "ga",
    "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
}
MULTI = [("new hampshire", "nh"), ("new jersey", "nj"), ("new mexico", "nm"), ("new york", "ny"),
         ("north carolina", "nc"), ("north dakota", "nd"), ("south carolina", "sc"),
         ("south dakota", "sd"), ("west virginia", "wv"), ("rhode island", "ri"),
         ("district of columbia", "dc"), ("tamil nadu", "tn"), ("andhra pradesh", "ap"),
         ("madhya pradesh", "mp"), ("uttar pradesh", "up"), ("himachal pradesh", "hp"),
         ("west bengal", "wb"), ("jammu and kashmir", "jk"), ("new delhi", "delhi"),
         ("greater mumbai", "mumbai"), ("hauts de france", "hdf"), ("ile de france", "idf")]
STREET = {
    "street": "st", "str": "st", "drive": "dr", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "lane": "ln", "court": "ct", "place": "pl", "circle": "cir",
    "terrace": "ter", "highway": "hwy", "parkway": "pkwy", "way": "wy", "trail": "trl",
    "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt",
    "suite": "ste", "building": "bldg", "floor": "flr", "fl": "flr", "room": "rm", "near": "nr",
    "opposite": "opp", "op": "opp", "nagar": "ngr", "colony": "col", "sector": "sec",
    "block": "blk", "village": "vill", "vil": "vill", "district": "dist", "r": "rue",
    "chemin": "che", "impasse": "imp", "allee": "all", "first": "1st", "second": "2nd",
    "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th", "seventh": "7th",
    "eighth": "8th", "ninth": "9th", "tenth": "10th", "mount": "mt", "saint": "st", "fort": "ft",
    "center": "ctr", "centre": "ctr", "township": "twp",
}
ADDR_DROP = {"no", "n/a", "na", "null", "unit", "door", "h", "hno", "house", "plot", "flat",
             "shop", "ste", "apt", "number", "num", "the", "of", "de", "du", "des", "la", "le"}

LEGAL_SK = {"prvt", "lmtd", "elp", "lp"}
_nonalnum = re.compile(r"[^a-z0-9]+")
_vowels = re.compile(r"[aeiouy]")
_rep = re.compile(r"(.)\1+")
_num = re.compile(r"\d+")


def ascii_lower(s):
    if not s:
        return ""
    if not s.isascii():
        s = unidecode(s)
    return s.lower()


def skel(tok):
    """Consonant skeleton: robust to transliteration / vowel noise / doubled letters."""
    t = tok.replace("ph", "f").replace("ck", "k").replace("q", "k").replace("z", "j").replace("w", "v")
    t = _rep.sub(r"\1", t)
    if len(t) > 1:
        t = t[0] + _vowels.sub("", t[1:])
    return _rep.sub(r"\1", t)


def clean_name(raw):
    s = ascii_lower(raw).replace("&", " and ").replace("+", " and ")
    s = re.sub(r"\.(com|net|org|in|co|fr|biz|info|us)\b", " ", s)
    s = s.replace("www.", " ")
    toks = []
    for t in _nonalnum.split(s):
        if not t:
            continue
        if not t.isdigit() and not t.isalpha():  # mixed alnum → leetspeak fix
            t = t.translate(LEET)
            t = "".join(c for c in t if c.isalpha()) or t
        toks.append(t)
    return toks


def clean_addr(raw):
    s = ascii_lower(raw)
    s = s.replace("<null>", " ").replace("n/a", " ")
    s = " " + _nonalnum.sub(" ", s) + " "
    for a, b in MULTI:
        if a in s:
            s = s.replace(" " + a + " ", " " + b + " ")
    toks = []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        elif t[0].isdigit():  # 016305a, 0110
            m = _num.match(t)
            t = (m.group().lstrip("0") or "0") + t[m.end():]
        t = STREET.get(t, t)
        t = STATES.get(t, t)
        if t in ADDR_DROP:
            continue
        toks.append(t)
    return toks


def process_rows(rows):
    out = []
    for eid, name, addr in rows:
        nt = clean_name(name)
        core = [t for t in nt if t not in LEGAL and skel(t) not in LEGAL_SK] or nt
        at = clean_addr(addr)
        sk = [skel(t) for t in core]
        nums = [t for t in at if t[0].isdigit()]
        hn = nums[0] if nums else ""
        alpha = [t for t in at if not t[0].isdigit()]
        coreflat = "".join(core)
        keys = set()
        for t in core:
            if len(t) >= 2:
                keys.add("n:" + t)
        for t in sk:
            if len(t) >= 2:
                keys.add("k:" + t)
        if len(coreflat) >= 4:
            keys.add("p:" + coreflat[:6])
            keys.add("q:" + skel(coreflat)[:5])
        for t in alpha:
            if len(t) >= 3:
                keys.add("a:" + t)
        for t in nums[:4]:
            keys.add("d:" + t)
        if hn and alpha:
            # house number + each street-ish word → very selective composite key
            for t in alpha[:6]:
                if len(t) >= 3:
                    keys.add("h:" + hn + "_" + t)
        out.append((eid, " ".join(nt), " ".join(core), " ".join(sk), " ".join(at), hn,
                    " ".join(alpha), " ".join(nums), list(keys)))
    return out


COLS = ["entity_id", "name", "core", "skel", "addr", "hn", "addr_alpha", "addr_nums", "keys"]


def process_df(rows):
    return pl.DataFrame(process_rows(rows), schema=COLS, orient="row")


def read_tsv(path):
    """Raw TSV -> DataFrame of strings. Quotes are not special in these files."""
    return pl.read_csv(path, separator="	", quote_char=None, infer_schema=False,
                       truncate_ragged_lines=True, encoding="utf8-lossy")


def run_one(df, is_reference, out_path):
    """Normalize one (split, source, country) slice and write it as Parquet."""
    t0 = time.time()
    names = df["business_name"].fill_null("").to_list()
    rows = list(zip(df["entity_id"].to_list(), names, df["business_address"].fill_null("").to_list()))
    # Source 1 is the Latin-script reference; for S2/S3 flag names that needed transliteration
    translit = [False] * len(names) if is_reference else [bool(x) and not x.isascii() for x in names]
    n = len(rows)
    step = 50_000
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        parts = pool.map(process_df, [rows[i:i + step] for i in range(0, n, step)], chunksize=1)
    del rows
    out = pl.concat(parts)
    del parts
    out = out.with_columns(pl.Series("translit", translit, dtype=pl.Boolean),
                           pl.int_range(0, n, dtype=pl.Int32).alias("idx"))
    out.write_parquet(out_path)
    print(f"  {out_path.name}: {n:,} rows ({time.time()-t0:.0f}s)", flush=True)


def main():
    splits = sys.argv[1:] or ["train", "test"]
    for split in splits:
        for k in (1, 2, 3):
            src = DATA / f"{split}_source{k}.tsv"
            if any(WORK.glob(f"{split}_s{k}_*.parquet")):  # already done
                continue
            df = read_tsv(src)
            print(f"{src.name}: {df.height:,} rows", flush=True)
            for country in sorted(df["country"].drop_nulls().unique().to_list()):
                run_one(df.filter(pl.col("country") == country), k == 1,
                        WORK / f"{split}_s{k}_{country}.parquet")
            del df


if __name__ == "__main__":
    main()
