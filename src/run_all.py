"""
run_all.py — End-to-end pipeline: raw TSVs -> blocking -> features -> model -> output files.

    python run_all.py            # full run (train + test), ~1.5 h on a 12-core / 16 GB laptop
    python run_all.py --predict  # skip training, use the shipped model in ../models/

Paths and settings come from config.py (override with ER_DATA / ER_WORK / ER_OUT etc.).
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(*args):
    t0 = time.time()
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], cwd=HERE, check=True)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


def main():
    predict_only = "--predict" in sys.argv
    splits = ["test"] if predict_only else ["train", "test"]
    step("prep.py", *splits)
    for sp in splits:
        step("block.py", sp)
    if not predict_only:
        step("eval_block.py")
    for sp in splits:
        step("features.py", sp)
    if not predict_only:
        step("train.py")
    step("predict.py")


if __name__ == "__main__":
    main()
