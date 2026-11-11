"""Aggregate comm_eval3 over judges: majority verdict, per-judge rates, agreement, objective leak rate, audit sheet."""
import csv
import glob
import itertools
import json
import os
import random
import sys

ROOT = "/data/yaokang/corrmem"
JUDGES = ["llama31", "qwen25-7b", "qwen3-8b"]
methods = sys.argv[1].split(",")
import analyze
analyze.KEY = "label_qwen25-7b"
VALID = analyze.valid_set("main3")[0]


def maj(vs):
    for v in "BAC":
        if vs.count(v) >= 2:
            return v
    return "B"             # three-way split: count against the memory (conservative)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in "ABC")
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


out, per = {}, {}
for m in methods:
    L = {j: json.load(open(f"{ROOT}/results/comm3_main3_{m}_{j}.json")) for j in JUDGES if os.path.exists(f"{ROOT}/results/comm3_main3_{m}_{j}.json")}
    if len(L) < 3:
        print("missing judges for", m, list(L)); continue
    n = len(L[JUDGES[0]])
    key = [(x["q"], x["uid"]) for x in L[JUDGES[0]]]
    rows = []
    for i in range(n):
        vs = [L[j][i]["v"] for j in JUDGES]
        vr = [L[j][i]["vref"] for j in JUDGES]
        x0 = L[JUDGES[0]][i]
        rows.append({"q": x0["q"], "uid": x0["uid"], "v": maj(vs), "vref": maj(vr), "vs": vs, "elig": x0["elig"], "leak": x0["leak"]})
    per[m] = rows
    pct = lambda f, R: 100 * sum(f(x) for x in R) / max(1, len(R))
    stale = lambda x: x["v"] == "B"
    missing = lambda x: x["vref"] == "A" and x["v"] == "C"
    sound = lambda x: not stale(x) and not missing(x)
    qs = sorted({x["q"] for x in rows})
    byq = {q: [x for x in rows if x["q"] == q] for q in qs}
    out[m] = {"units": n, "questions": len(qs), "stale": pct(stale, rows), "missing": pct(missing, rows), "sound": pct(sound, rows),
              "q_sound": 100 * sum(all(sound(x) for x in byq[q]) for q in qs) / len(qs),
              "q_stale_free": 100 * sum(all(not stale(x) for x in byq[q]) for q in qs) / len(qs),
              "updated": pct(lambda x: x["v"] == "A", rows),
              "leak": pct(lambda x: x["leak"], [x for x in rows if x["elig"]]), "leak_n": sum(x["elig"] for x in rows),
              "per_judge_stale": {j: 100 * sum(L[j][i]["v"] == "B" for i in range(n)) / n for j in JUDGES}}
    v = [x["q"] in VALID for x in rows]
    out[m]["q_sound_valid"] = 100 * sum(all(sound(x) for x in byq[q]) for q in qs if q in VALID) / max(1, sum(q in VALID for q in qs))
    if m == methods[0]:
        for j1, j2 in itertools.combinations(JUDGES, 2):
            out.setdefault("_kappa", {})[f"{j1}~{j2}"] = kappa([x["v"] for x in L[j1]], [x["v"] for x in L[j2]])
# agreement of the surface leak with the verdict (does the judge-free check corroborate it?)
allrows = [x for m in per for x in per[m] if x["elig"]]
out["_leak_vs_verdict"] = {"P(leak|B)": 100 * sum(x["leak"] for x in allrows if x["v"] == "B") / max(1, sum(x["v"] == "B" for x in allrows)),
                           "P(leak|A)": 100 * sum(x["leak"] for x in allrows if x["v"] == "A") / max(1, sum(x["v"] == "A" for x in allrows)),
                           "P(leak|C)": 100 * sum(x["leak"] for x in allrows if x["v"] == "C") / max(1, sum(x["v"] == "C" for x in allrows))}
json.dump(out, open(f"{ROOT}/results/eval3.json", "w"), indent=1)
json.dump({m: per[m] for m in per}, open(f"{ROOT}/results/eval3_units.json", "w"))
print(json.dumps(out, indent=1))
# audit sheet: 120 units stratified over methods, texts and the three verdicts, for manual labelling
rnd = random.Random(0)
sheet = []
cands = [m for m in ["append", "coarse", "rebuild", "recompress", "comm10", "comm11"] if m in per]
for m in cands:
    for x in rnd.sample(per[m], min(20, len(per[m]))):
        sheet.append((m, x))
with open(f"{ROOT}/results/audit_sheet.csv", "w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["method", "qid", "uid", "judge_llama", "judge_qwen25", "judge_qwen3", "majority", "old_claim", "correction_message", "unit_before", "unit_after", "human_label(A/B/C)"])
    corr = {r["qid"]: r for r in json.load(open(f"{ROOT}/data/lme_corr.json"))}
    for m, x in sheet:
        d = json.load(open(glob.glob(f"{ROOT}/results/main3/{m}/{x['q']}.json")[0]))
        st = json.load(open(f"{ROOT}/states/cocomem/{x['q']}.json"))
        ref = json.load(open(glob.glob(f"{ROOT}/results/main3/recompress/{x['q']}.json")[0]))
        old = " ".join(st["claims"][c]["text"] for c in ref["repair"]["S1"]["revised"] if c in st["claims"])
        w.writerow([m, x["q"], x["uid"], *x["vs"], x["v"], old, corr[x["q"]]["affect"]["correction_message"],
                    d["units"]["S0"][x["uid"]], d["units"]["S1"][x["uid"]], ""])
print("audit rows", len(sheet))
