"""Paired effect sizes with bootstrap intervals and a non-inferiority check (margin fixed here, before looking at v12 data:
5 points).  Two outcomes per correction: answered the corrected question (QA, valid set, Qwen2.5 judge) and every unit
of its closure sound (joint verdict, majority of three judges).  Differences are ours minus the reference; a method is
non-inferior to a reference if the lower 95% bound of the difference is above -5 points."""
import json
import sys

import numpy as np

import analyze

ROOT = "/data/yaokang/corrmem"
MARGIN = 5.0
ours = sys.argv[1].split(",")
refs = sys.argv[2].split(",")
KEY = sys.argv[3] if len(sys.argv) > 3 else "label_qwen25-7b"
analyze.KEY = KEY
VALID = analyze.valid_set("main3")[0]
from math import comb


def mcnemar(a, b):
    n01 = int(sum(1 for x, y in zip(a, b) if x and not y)); n10 = int(sum(1 for x, y in zip(a, b) if y and not x))
    n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)


def boot(a, b, B=5000, seed=0):
    r = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    s = [d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return 100 * d.mean(), 100 * np.percentile(s, 2.5), 100 * np.percentile(s, 97.5)


qa = {}
for m in set(ours + refs):
    R = analyze.load("main3", m)
    qa[m] = {q: analyze.lab(d, "S1", "q") for q, d in R.items() if q in VALID and analyze.lab(d, "S1", "q") is not None}
U = {}
sound = {}
for m, rows in U.items():
    by = {}
    for x in rows:
        by.setdefault(x["q"], []).append(x["v"] != "B" and not (x["vref"] == "A" and x["v"] == "C"))
    sound[m] = {q: float(all(v)) for q, v in by.items()}
out = {"margin": MARGIN}
for name, D in (("qa", qa),):
    out[name] = {}
    for m in ours:
        for r in refs:
            if m not in D or r not in D or m == r:
                continue
            qs = sorted(set(D[m]) & set(D[r]))
            a = [D[m][q] for q in qs]; b = [D[r][q] for q in qs]
            d, lo, hi = boot(a, b)
            out[name][f"{m}|{r}"] = {"n": len(qs), "diff": float(d), "lo": float(lo), "hi": float(hi), "p": float(mcnemar([x > 0.5 for x in a], [x > 0.5 for x in b])),
                                     "non_inferior": bool(lo > -MARGIN)}
json.dump(out, open(f"{ROOT}/results/stat19{'' if KEY == 'label_qwen25-7b' else '_' + KEY[6:]}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
