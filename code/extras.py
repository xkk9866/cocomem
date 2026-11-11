"""Reviewer-requested analyses (CPU only):
 1. filtering flow of LongMemEval-Corr (generated -> in test set -> oracle-valid), per question type
 2. cost scaling: tokens / latency per correction vs history size (#claims, raw tokens) and vs affected units
 3. accuracy-cost frontier points
 4. LoCoMo statistics with conversation-level cluster bootstrap
 5. judge agreement (Cohen's kappa) between the three judges
Writes results/extras_<tag>.json"""
import argparse, glob, json, os
import numpy as np

ROOT = "/data/yaokang/corrmem"


def load(tag, m):
    return {json.load(open(f)).get("qid") or json.load(open(f)).get("cid"): json.load(open(f))
            for f in glob.glob(f"{ROOT}/results/{tag}/{m}/*.json")}


def lab(d, sc, qk, key):
    for a in d["answers"]:
        if a["scenario"] == sc and a["qkey"] == qk and key in a:
            return float(a[key])
    return None


def kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main(a):
    out = {}
    corr = {r["qid"]: r for r in json.load(open(f"{ROOT}/data/lme_corr.json"))}
    sp = json.load(open(f"{ROOT}/data/lme_splits.json"))
    test = sp["test200"]
    o = load(a.tag, "oracle")
    key = a.key
    flow = {"test_questions": len(test),
            "with_evidence_correction": sum("affect" in corr.get(q, {}) for q in test),
            "with_unrelated_correction": sum("unrel" in corr.get(q, {}) for q in test),
            "oracle_valid_evidence": sum(lab(o[q], "S1", "q", key) == 1.0 for q in test if q in o and "affect" in corr.get(q, {})),
            "oracle_valid_probe": sum(lab(o[q], "S2", "probe", key) == 1.0 for q in test if q in o and "unrel" in corr.get(q, {}))}
    out["flow"] = flow
    # cost scaling
    sc = {}
    for m in a.methods.split(","):
        R = load(a.tag, m)
        pts = []
        for q, d in R.items():
            st = f"{ROOT}/states/cocomem/{q}.json"
            if not os.path.exists(st):
                continue
            n_claims = len(json.load(open(st))["claims"])
            for s in ("S1", "S2"):
                r = d.get("repair", {}).get(s)
                if r and "usage" in r:
                    pts.append({"claims": n_claims, "tok": r["usage"]["in"] + r["usage"]["out"],
                                "time": r.get("time", 0), "closure": len(r.get("closure", []))})
        if not pts:
            continue
        c = np.array([p["claims"] for p in pts]); t = np.array([p["tok"] for p in pts])
        bins = np.quantile(c, [0, .25, .5, .75, 1])
        sc[m] = {"n": len(pts), "tok_mean": float(t.mean()), "time_mean": float(np.mean([p["time"] for p in pts])),
                 "corr_claims_tok": float(np.corrcoef(c, t)[0, 1]) if t.std() > 0 else 0.0,
                 "slope_tok_per_1k_claims": float(np.polyfit(c / 1000, t, 1)[0]) if t.std() > 0 else 0.0,
                 "by_quartile": [float(t[(c >= bins[i]) & (c <= bins[i + 1])].mean()) for i in range(4)],
                 "quartile_edges": [float(x) for x in bins],
                 "closure_mean": float(np.mean([p["closure"] for p in pts]))}
    out["scaling"] = sc
    # LoCoMo cluster bootstrap on corrected-answer accuracy
    lc = {}
    rng = np.random.default_rng(0)
    for m in a.locomo.split(","):
        R = load("locomo", m)
        conv = []
        for cid, d in R.items():
            v = [float(x[key]) for x in d["answers"] if x["scenario"] == "S1" and x["qkey"].startswith("a") and key in x]
            h = [float(x[key]) for x in d["answers"] if x["scenario"] == "S1" and x["qkey"].startswith("h") and key in x]
            if v:
                conv.append((v, h))
        if not conv:
            continue
        def stat(cs, j):
            return 100 * np.mean([x for c in cs for x in c[j]])
        bs = [[stat([conv[i] for i in rng.integers(0, len(conv), len(conv))], j) for _ in range(2000)] for j in (0, 1)]
        lc[m] = {"corr": stat(conv, 0), "corr_ci": [float(np.percentile(bs[0], 2.5)), float(np.percentile(bs[0], 97.5))],
                 "hist": stat(conv, 1), "hist_ci": [float(np.percentile(bs[1], 2.5)), float(np.percentile(bs[1], 97.5))]}
    out["locomo_cluster"] = lc
    # judge agreement
    keys = ["label_qwen3-8b", "label_qwen25-7b", "label_llama31"]
    pairs = {}
    vals = {k: [] for k in keys}
    for f in glob.glob(f"{ROOT}/results/{a.tag}/*/*.json"):
        for x in json.load(open(f))["answers"]:
            if all(k in x for k in keys):
                for k in keys:
                    vals[k].append(float(x[k]))
    for i in range(3):
        for j in range(i + 1, 3):
            if vals[keys[i]]:
                pairs[f"{keys[i]}~{keys[j]}"] = {"kappa": kappa(vals[keys[i]], vals[keys[j]]),
                                                 "agree": float(np.mean(np.array(vals[keys[i]]) == np.array(vals[keys[j]])))}
    out["judge_agreement"] = pairs
    out["judge_n"] = len(vals[keys[0]])
    json.dump(out, open(f"{ROOT}/results/extras_{a.tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main3")
    ap.add_argument("--key", default="label_qwen25-7b")
    ap.add_argument("--methods", default="ccf,slots,coarse,self_span,recompress,spec,rebuild,simupdate,append")
    ap.add_argument("--locomo", default="ccf,slots,spec,coarse,recompress,rebuild,append,simupdate,self,rawrag,membank,mem0,amem")
    main(ap.parse_args())
