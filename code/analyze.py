"""Aggregate judged results into the evaluation tables (LongMemEval-Corr and LoCoMo-Corr)."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

DIRECT = ("single-session-user", "single-session-assistant", "knowledge-update")
DERIVED = ("multi-session", "temporal-reasoning")
ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem") + "/results"
KEY = None   # judge label key; None -> "label" (latest judge)
CANON = {}   # qid -> canonical S0 record shared by all strategies on the same CoCoMem memory
LEDGER = {"append", "slots", "simupdate", "self", "self_span", "coarse", "ncf", "ccf", "recompress", "spec",
          "spec_ncf", "spec_overwrite", "ccf_overwrite", "rebuild", "comm", "comm_zs", "comm_overwrite", "comm2",
          "comm2_overwrite", "comm3", "comm3_overwrite", "comm3_nofallback", "comm4", "comm4_overwrite", "comm5", "comm5_overwrite", "comm6", "comm7", "comm7_overwrite", "comm8", "comm10", "comm10_overwrite", "comm11", "comm12", "comm12_overwrite", "comm12L", "comm12T", "comm12F", "comm13", "comm13L", "comm13T",
          "comm20", "comm20_nw", "comm20_cite", "comm20_nv", "comm20_nr", "comm20_overwrite", "comm20_zs",
          "comm21", "comm21_sel", "comm21_nd", "comm21_w", "comm21_nv", "comm21_nr", "comm21_overwrite", "comm21_zs",
          "comm21_s1", "comm21_s2"}
QIDS = None      # optional restriction to a question set (blocks of partially finished official baselines)


def load(tag, method):
    out = {}
    for f in glob.glob(f"{ROOT}/{tag}/{method}/*.json"):
        d = json.load(open(f))
        k = d.get("qid") or d.get("cid")
        if QIDS is None or k in QIDS or method == "oracle":
            out[k] = d
    return out


def lab(d, sc, qk):
    k = KEY or "label"
    for a in d["answers"]:
        if a["scenario"] == sc and a["qkey"] == qk and k in a:
            return float(a[k])
    return None


VALID_KEY = None  # judge that defines the valid set (None: the reporting judge); fixed-set comparisons


def valid_set(tag, oracle="oracle"):
    """Corrections whose new answer is recoverable from evidence + correction by the oracle reader."""
    global KEY
    o = load(tag, oracle)
    if VALID_KEY:
        k0, KEY = KEY, VALID_KEY
        try:
            return {q for q, d in o.items() if lab(d, "S1", "q") == 1.0}, {q for q, d in o.items() if lab(d, "S2", "probe") == 1.0}
        finally:
            KEY = k0
    return {q for q, d in o.items() if lab(d, "S1", "q") == 1.0}, {q for q, d in o.items() if lab(d, "S2", "probe") == 1.0}


def mcnemar(a, b):
    """Exact two-sided McNemar p-value for paired binary outcomes."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def ci(x, B=2000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    s = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return 100 * np.percentile(s, 2.5), 100 * np.percentile(s, 97.5)


def summarize(tag, methods, restrict=True, ref=None):
    v1, v2 = valid_set(tag) if restrict else (None, None)
    CANON.update(load(tag, "recompress"))
    common = None
    loaded = {m: load(tag, m) for m in methods}
    for m, R in loaded.items():
        if R:
            common = set(R) if common is None else common & set(R)
    rows, per = [], {}
    for m in methods:
        R = loaded[m]
        if not R:
            continue
        acc, cost, paired = defaultdict(list), defaultdict(list), {}
        for q in sorted(common):
            d = R[q]
            t = d["qtype"]
            s0 = lab(CANON[q], "S0", "q") if (m in LEDGER and q in CANON) else lab(d, "S0", "q")
            if s0 is not None:
                acc["orig"].append(s0)
            if v1 is None or q in v1:
                x = lab(d, "S1", "q")
                if x is not None:
                    acc["corr"].append(x)
                    paired[q] = x
                    acc["corr_direct" if t in DIRECT else "corr_derived"].append(x)
                    if s0 == 1.0:
                        acc["fid"].append(x)
                h = lab(d, "S1", "hist")
                if h is not None:
                    acc["hist"].append(h)
                    if x is not None:
                        acc["both"].append(float(x == 1.0 and h == 1.0))
            x = lab(d, "S2", "q")
            if x is not None:
                acc["keep"].append(x)
            if v2 is None or q in v2:
                x = lab(d, "S2", "probe")
                if x is not None:
                    acc["probe"].append(x)
            for sc in ("S1", "S2"):
                r = d.get("repair", {}).get(sc)
                if r and "usage" in r:
                    cost["calls"].append(r["usage"]["calls"])
                    cost["tok"].append(r["usage"]["in"] + r["usage"]["out"])
                    cost["out"].append(r["usage"]["out"])
                    cost["closure"].append(len(r.get("closure", r.get("repaired", []))))
                    t = r.get("t_usage") or {"in": 0, "out": 0}      # distilled commutator (student) tokens
                    cost["ttok"].append(t["in"] + t["out"])
                    cost["time"].append(r.get("time", 0.0))
        row = {"method": m, "n": len(common)}
        for k in ("orig", "corr", "corr_direct", "corr_derived", "fid", "hist", "both", "keep", "probe"):
            row[k] = (100 * np.mean(acc[k]), len(acc[k])) if acc[k] else (float("nan"), 0)
        row["corr_ci"] = ci(acc["corr"]) if acc["corr"] else (float("nan"),) * 2
        for k in ("calls", "tok", "out", "closure", "ttok", "time"):
            row["c_" + k] = float(np.mean(cost[k])) if cost[k] else 0.0
        rows.append(row)
        per[m] = paired
    if ref and ref in per:
        for row in rows:
            qs = sorted(set(per[ref]) & set(per[row["method"]]))
            row["p_vs_ref"] = mcnemar([per[ref][q] for q in qs], [per[row["method"]][q] for q in qs])
    return rows


def show(rows):
    keys = ["orig", "corr", "corr_direct", "corr_derived", "fid", "hist", "both", "keep", "probe"]
    print(f"{'method':14s} {'n':>4s} " + " ".join(f"{k:>12s}" for k in keys) + f" {'CI':>13s} {'p':>6s} {'calls':>6s} {'tok':>7s} {'out':>6s} {'ttok':>6s} {'sec':>5s}")
    for r in rows:
        print(f"{r['method']:14s} {r['n']:4d} " + " ".join(f"{r[k][0]:6.1f}({r[k][1]:3d})" for k in keys)
              + f" [{r['corr_ci'][0]:4.1f},{r['corr_ci'][1]:4.1f}] {r.get('p_vs_ref', float('nan')):6.3f}"
              + f" {r['c_calls']:6.1f} {r['c_tok']:7.0f} {r['c_out']:6.0f} {r['c_ttok']:6.0f} {r['c_time']:5.1f}")


def locomo(methods):
    """LoCoMo-Corr: a* = corrected QA (S1 -> new answer), h* = history, u* = unaffected QA."""
    print(f"{'method':14s} {'orig':>6s} {'corr':>6s} {'hist':>6s} {'keep':>6s} {'both':>6s} {'tok/corr':>9s} {'n_corr':>6s}")
    out = []
    for m in methods:
        R = load("locomo", m)
        if not R:
            continue
        acc = defaultdict(list)
        toks, ttoks, secs, nc = [], [], [], 0
        canon = load("locomo", "recompress") if m in LEDGER else {}
        for cid, d in R.items():
            lab_ = {(a["scenario"], a["qkey"]): float(a[KEY or "label"]) for a in d["answers"] if (KEY or "label") in a}
            if cid in canon:
                for a in canon[cid]["answers"]:
                    if a["scenario"] == "S0" and (KEY or "label") in a:
                        lab_[("S0", a["qkey"])] = float(a[KEY or "label"])
            for (sc, k), v in lab_.items():
                if sc == "S0" and k.startswith("u"):
                    acc["orig"].append(v)
                if sc == "S1" and k.startswith("a"):
                    acc["corr"].append(v)
                    h = lab_.get(("S1", "h" + k[1:]))
                    if h is not None:
                        acc["both"].append(float(v == 1.0 and h == 1.0))
                if sc == "S1" and k.startswith("h"):
                    acc["hist"].append(v)
                if sc == "S1" and k.startswith("u"):
                    acc["keep"].append(v)
            st = d["repair"]["S1"]
            n = len(st.get("steps", [])) or 1
            nc += n
            toks.append((st["usage"]["in"] + st["usage"]["out"]) / n)
            tu = [s.get("t_usage") or {"in": 0, "out": 0} for s in st.get("steps", [])]
            ttoks.append(sum(t["in"] + t["out"] for t in tu) / n)
            secs.append(st.get("time", 0.0) / n)
        row = {k: 100 * np.mean(v) for k, v in acc.items()}
        row["tok"] = float(np.mean(toks)) if toks else 0
        row["ttok"], row["sec"] = float(np.mean(ttoks)) if ttoks else 0, float(np.mean(secs)) if secs else 0
        row["method"], row["n_corr"] = m, nc
        out.append(row)
        print(f"{m:14s} " + " ".join(f"{row.get(k, float('nan')):6.1f}" for k in ("orig", "corr", "hist", "keep", "both"))
              + f" {row['tok']:9.0f} {nc:6d} {row['ttok']:6.0f} {row['sec']:5.1f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--methods", default="")
    ap.add_argument("--ref", default="ncf")
    ap.add_argument("--key", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--valid_key", default=None)
    ap.add_argument("--qids", default="")
    a = ap.parse_args()
    KEY, VALID_KEY = a.key, a.valid_key
    QIDS = set(a.qids.split(",")) if a.qids else None
    ms = a.methods.split(",") if a.methods else sorted(os.listdir(f"{ROOT}/{a.tag}"))
    if a.tag == "locomo":
        rows = locomo(ms)
    else:
        rows = summarize(a.tag, ms, restrict=not a.all, ref=a.ref)
        show(rows)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=float)
