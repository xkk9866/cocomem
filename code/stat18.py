"""Paired tests on the direct correctness measure (correction level: all units of the closure correct)."""
import json, glob, itertools
from math import comb
R = "/data/yaokang/corrmem/results"
import analyze
analyze.KEY = "label_qwen25-7b"
VALID = analyze.valid_set("main3")[0]
def load(m):
    res = json.load(open(f"{R}/comm2_main3_{m}.json"))
    by = {}
    for x in res:
        if "uid" in x and x["q"] in VALID:
            by.setdefault(x["q"], []).append(x)
    return by
def mc(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
out = {}
ref = {m: load(m) for m in ["comm10", "comm11", "comm10_overwrite", "comm5", "comm6", "comm7", "comm8", "ncf", "coarse", "rebuild", "recompress", "comm_zs", "self_span"]}
qs = sorted(set.intersection(*[set(v) for v in ref.values()]))
for m, by in ref.items():
    out[m] = {"q_correct": 100 * sum(all(x["correct"] for x in by[q]) for q in qs) / len(qs),
              "unit_correct": 100 * sum(x["correct"] for q in qs for x in by[q]) / sum(len(by[q]) for q in qs), "n": len(qs)}
for a in ["comm10", "comm11"]:
    for b in ["ncf", "coarse", "rebuild", "comm_zs"] + ["comm5", "comm6", "comm7", "comm8", "recompress"] + (["comm10"] if a == "comm11" else []):
        A = [all(x["correct"] for x in ref[a][q]) for q in qs]; B = [all(x["correct"] for x in ref[b][q]) for q in qs]
        out[f"p_{a}_vs_{b}"] = mc(A, B)
json.dump(out, open(f"{R}/stat18.json", "w"), indent=1); print(json.dumps(out, indent=1))
