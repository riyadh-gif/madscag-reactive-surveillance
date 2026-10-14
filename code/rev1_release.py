# ══════════════════════════════════════════════════════════════
# Build the public release tree, with village identity removed
# ══════════════════════════════════════════════════════════════
"""Assembles everything the reviewers asked to be released, and nothing
that would identify a village.

Six stored files carry the plain village key `kab|kec|desa`. They are the
join contract for every tensor and prediction array, so they cannot simply
be dropped; they are rewritten with a salted hash in place of the key, the
three name columns removed, and the regency reduced to an integer code so
that leave-one-regency-out stays reproducible. Row order is preserved and
checked, because position is what ties these files to the `.npy` and
`.npz` arrays.

The salt is generated once and written outside the release tree. It is
never committed: without it the hash cannot be reversed by matching
against the public list of Indonesian village names, which is small enough
to enumerate.

The script refuses to finish if any identity column, any plain village key
or any machine-specific path survives in the tree it just built.
"""
import hashlib
import json
import os
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(os.environ["MADSCAG_WORK"])
CODE = Path(__file__).parent
ROOT = WORK.parent.parent
RELEASE = Path(os.environ.get("MADSCAG_RELEASE", ROOT / "release_madscag"))
SALT_FILE = WORK / "release_salt.txt"          # never inside RELEASE

PREP = WORK / "preprocessed_v7"
EVAL = WORK / "eval_v7"
EXP = WORK / "experiments_v7"

ID_COLS = ["kab", "kec", "desa"]
NAME_RE = r"^[a-z]+\|[a-z]+\|[a-z]+"           # the plain vid shape
SKIP_CODE = {"__pycache__"}
SKIP_SUFFIX = {".bak", ".rev1bak", ".pyc"}
# runs whose weights are excluded; predictions and metrics still ship
WEIGHT_FILES = {"final.keras"}
# a dated per-village predicted-risk list. Hashing the key does not make it
# publishable: it names which villages a model flagged, one row each, and the
# aggregate versions in revisi1 carry everything the manuscript reports.
EXCLUDE_EVAL = {"watchlist_v7_tuned.csv"}


def log(m):
    print(m, flush=True)


def get_salt():
    if SALT_FILE.exists():
        return SALT_FILE.read_text(encoding="utf-8").strip()
    salt = secrets.token_hex(32)
    SALT_FILE.write_text(salt + "\n", encoding="utf-8")
    log(f"generated a new salt at {SALT_FILE} (keep it out of the repository)")
    return salt


def hash_vid(vid, salt):
    return hashlib.sha256((salt + "|" + vid).encode("utf-8")).hexdigest()[:16]


def kab_codes(all_kab):
    return {k: i + 1 for i, k in enumerate(sorted(all_kab))}


def deidentify(df, salt, codes):
    """Replace the village key, drop the names, code the regency."""
    out = df.copy()
    if "vid" in out.columns:
        out["vid_hash"] = [hash_vid(v, salt) for v in out["vid"]]
        out = out.drop(columns=["vid"])
    if "kab" in out.columns:
        out["kab_code"] = out["kab"].map(codes).astype("Int64")
    out = out.drop(columns=[c for c in ID_COLS if c in out.columns])
    first = [c for c in ["vid_hash", "kab_code", "t"] if c in out.columns]
    return out[first + [c for c in out.columns if c not in first]]


def copy_code():
    dst = RELEASE / "code"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "figures").mkdir(exist_ok=True)
    n = 0
    for f in sorted(CODE.iterdir()):
        if f.is_dir() or f.suffix in SKIP_SUFFIX or f.name in SKIP_CODE:
            continue
        if f.suffix not in {".py", ".sh"}:
            continue
        target = dst / "figures" / f.name if f.name.startswith("make_fig") else dst / f.name
        shutil.copy2(f, target)
        n += 1
    return n


def copy_predictions():
    dst = RELEASE / "artifacts" / "predictions"
    dst.mkdir(parents=True, exist_ok=True)
    runs = n_files = 0
    for run in sorted(EXP.iterdir()):
        if not run.is_dir():
            continue
        d = dst / run.name
        d.mkdir(exist_ok=True)
        for f in run.iterdir():
            if f.name in WEIGHT_FILES or f.suffix in SKIP_SUFFIX:
                continue
            if f.suffix == ".npz":
                z = np.load(f)
                np.savez_compressed(d / f.name, **{k: z[k] for k in z.files})
            else:
                shutil.copy2(f, d / f.name)
            n_files += 1
        runs += 1
    return runs, n_files


def copy_eval(salt, codes):
    dst = RELEASE / "artifacts" / "eval"
    (dst / "revisi1").mkdir(parents=True, exist_ok=True)
    n = 0
    for src_dir, out_dir in [(EVAL, dst), (EVAL / "revisi1", dst / "revisi1")]:
        for f in sorted(src_dir.iterdir()):
            if f.is_dir() or f.suffix not in {".csv", ".json"}:
                continue
            if f.name in EXCLUDE_EVAL:
                log(f"  withheld {f.name} (per-village risk list)")
                continue
            if f.suffix == ".csv":
                df = pd.read_csv(f)
                if any(c in df.columns for c in ID_COLS + ["vid"]):
                    before = len(df)
                    df = deidentify(df, salt, codes)
                    assert len(df) == before, f"row count changed in {f.name}"
                    df.to_csv(out_dir / f.name, index=False)
                    log(f"  de-identified {f.name} ({before} rows)")
                else:
                    shutil.copy2(f, out_dir / f.name)
            else:
                shutil.copy2(f, out_dir / f.name)
            n += 1
    return n


def build_data(salt, codes):
    """Only what carries no register row and no village ordering.

    Row-keyed files are deliberately absent: replacing the village key with a
    hash is not sufficient protection for them, so the split indices, the
    derived panel and the gap table travel with the data bundle under the
    terms of the original data request and are supplied on request.

    What ships instead is the period membership of each split, which is what
    the reviewer's request for the exact split period indices asks for.
    """
    dst = RELEASE / "data"
    dst.mkdir(parents=True, exist_ok=True)
    written = []
    rows = []
    for sp in ["latih", "sahih", "uji"]:
        t = pd.read_csv(PREP / f"sample_meta_{sp}.csv", usecols=["t"]).t
        for v in sorted(t.unique()):
            rows.append(dict(split=sp, t=int(v), n_windows=int((t == v).sum())))
    periods = pd.DataFrame(rows)
    periods.to_csv(dst / "split_periods.csv", index=False)
    written.append(("split_periods.csv", len(periods)))

    shutil.copy2(PREP / "meta_v7.json", dst / "meta_v7.json")
    shutil.copy2(PREP / "scaler_X_v7.pkl", dst / "scaler_X_v7.pkl")
    written.append(("meta_v7.json", 0))
    written.append(("scaler_X_v7.pkl", 0))
    return written


def verify():
    """Refuse to hand over a tree that still identifies anything."""
    import re
    bad = []
    pat = re.compile(NAME_RE)
    for f in RELEASE.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix == ".csv":
            head = pd.read_csv(f, nrows=200)
            for c in ID_COLS + ["vid"]:
                if c in head.columns:
                    bad.append(f"{f.relative_to(RELEASE)}: identity column '{c}'")
            for c in head.columns:
                if head[c].dtype == object and head[c].astype(str).str.match(pat).any():
                    bad.append(f"{f.relative_to(RELEASE)}: plain village key in '{c}'")
            # a hashed key is no safer here: the stored row order is an
            # alphabetical sort of the village name, so any row-keyed file
            # aligns to the public village list regardless of the hash
            if "vid_hash" in head.columns:
                bad.append(f"{f.relative_to(RELEASE)}: row-keyed by village "
                           "('vid_hash'); this belongs to the data bundle, "
                           "not the repository")
        if f.suffix in {".py", ".sh", ".md", ".json", ".txt"}:
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # assembled from fragments so this file does not match itself;
            # the venv directory name is a local layout detail, not a
            # disclosure, and the runners now take MADSCAG_PY instead
            needles = ["Magister" + " PENS", "journal" + "-arthur",
                       "C:" + chr(92) + "Users", "/ho" + "me/", "/Us" + "ers/"]
            for needle in needles:
                if needle in t:
                    bad.append(f"{f.relative_to(RELEASE)}: path leak '{needle}'")
    return bad


def main():
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)
    salt = get_salt()
    codes = kab_codes(pd.read_csv(PREP / "sample_meta_uji.csv", usecols=["kab"]).kab.unique())
    log(f"regency codes: {codes}")

    n_code = copy_code()
    log(f"code: {n_code} files")
    runs, n_pred = copy_predictions()
    log(f"predictions: {runs} runs, {n_pred} files (weights excluded)")
    n_eval = copy_eval(salt, codes)
    log(f"eval artifacts: {n_eval} files")
    for name, rows in build_data(salt, codes):
        log(f"data: {name} ({rows} rows)")

    (RELEASE / "docs").mkdir(exist_ok=True)
    (RELEASE / "figures").mkdir(exist_ok=True)

    bad = verify()
    if bad:
        log("\nREFUSING TO PROCEED, the tree still discloses:")
        for b in bad[:40]:
            log("  " + b)
        sys.exit(1)

    total = sum(f.stat().st_size for f in RELEASE.rglob("*") if f.is_file())
    big = [(f.relative_to(RELEASE), f.stat().st_size / 1e6)
           for f in RELEASE.rglob("*") if f.is_file() and f.stat().st_size > 50e6]
    log(f"\nclean. total {total/1e6:.1f} MB")
    log(f"files above 50 MB: {big if big else 'none'}")
    log(f"salt kept at {SALT_FILE}, which must stay out of the repository")


if __name__ == "__main__":
    main()
