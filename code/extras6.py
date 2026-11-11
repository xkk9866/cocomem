"""v5 analyses: cost vs history size (8B and commutator tokens, latency), cost vs position in a sequence
of corrections (LoCoMo), conversation-clustered CIs, three-judge agreement, edit statistics."""
import glob
import json
import os

import numpy as np

from extras import kappa, load

ROOT = "/data/yaokang/corrmem"
LME = (os.environ.get("X_LME") or "comm10,comm11,comm10_overwrite,comm5,comm6,comm7,comm8,comm4,append,slots,simupdate,self_span,coarse,ncf,ccf,recompress,spec,rebuild,comm_zs,comm,comm2,comm3").split(",")
LOC = (os.environ.get("X_LOC") or "comm10,comm11,comm10_overwrite,comm5,comm6,comm7,comm8,comm4,comm4_overwrite,comm3,comm3_nofallback,comm3_overwrite,comm2,comm2_overwrite,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem").split(",")
KEY = "label_qwen25-7b"


def tok(u):
    return (u or {}).get("in", 0) + (u or {}).get("out", 0)


def main():
    out = {}
    nclaims = {}
    for f in glob.glob(f"{ROOT}/states/cocomem/*.json"):
        d = json.load(open(f))
        nclaims[os.path.basename(f)[:-5]] = (len(d["claims"]), sum(len(c["text"]) for c in d["claims"].values()) // 4)
    sc = {}
    for m in LME:
        pts = []
        for q, d in load("main3", m).items():
            if q not in nclaims:
                continue
            for s in ("S1", "S2"):
                r = d.get("repair", {}).get(s)
                if r and "usage" in r:
                    pts.append((nclaims[q][0], nclaims[q][1], tok(r["usage"]), tok(r.get("t_usage")), r.get("time", 0.0),
                                len(r.get("closure", [])), r.get("closure_sents", 0)))
        P = np.array(pts, float)
        edges = np.quantile(P[:, 0], [0, .25, .5, .75, 1])
        qbin = lambda col: [float(P[(P[:, 0] >= edges[i]) & (P[:, 0] <= edges[i + 1]), col].mean()) for i in range(4)]
        sc[m] = {"n": len(P), "tok": float(P[:, 2].mean()), "ttok": float(P[:, 3].mean()), "sec": float(P[:, 4].mean()),
                 "tok_by_q": qbin(2), "ttok_by_q": qbin(3), "edges": [float(x) for x in edges],
                 "corr_claims_tok": float(np.corrcoef(P[:, 0], P[:, 2])[0, 1]) if P[:, 2].std() > 0 else 0.0,
                 "corr_claims_ttok": float(np.corrcoef(P[:, 0], P[:, 3])[0, 1]) if P[:, 3].std() > 0 else 0.0,
                 "closure_units": float(P[:, 5].mean()), "closure_sents": float(P[:, 6].mean()),
                 "hist_tokens_mean": float(np.mean([v[1] for v in nclaims.values()]))}
    out["scaling"] = sc
    # sequential corrections on LoCoMo: cost of the i-th correction
    seq = {}
    pos = {}
    for m in (os.environ.get("X_SEQ") or "comm10,comm11,comm4,comm3,comm2,ncf,coarse,recompress,spec,rebuild,self").split(","):
        by, acc = {}, {}
        for cid, d in load("locomo", m).items():
            for i, st in enumerate(d["repair"]["S1"].get("steps", [])):
                by.setdefault(i // 4, []).append((tok(st.get("usage")), tok(st.get("t_usage"))))
            for x in d["answers"]:
                if x["scenario"] == "S1" and x["qkey"].startswith("a") and KEY in x:
                    acc.setdefault(min(int(x["qkey"][1:]) // 4, 2), []).append(float(x[KEY]))
        seq[m] = {str(k): [float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))] for k, v in sorted(by.items())}
        pos[m] = {str(k): [100 * float(np.mean(v)), len(v)] for k, v in sorted(acc.items())}
    out["locomo_sequential"] = seq
    out["locomo_position"] = pos
    # conversation-clustered bootstrap on LoCoMo
    rng = np.random.default_rng(0)
    lc = {}
    for m in LOC:
        conv = []
        for cid, d in load("locomo", m).items():
            g = lambda p: [float(x[KEY]) for x in d["answers"] if x["scenario"] == "S1" and x["qkey"].startswith(p) and KEY in x]
            if g("a"):
                conv.append((g("a"), g("h")))
        if not conv:
            continue
        stat = lambda cs, j: 100 * np.mean([x for c in cs for x in c[j]])
        bs = [[stat([conv[i] for i in rng.integers(0, len(conv), len(conv))], j) for _ in range(2000)] for j in (0, 1)]
        lc[m] = {"corr": stat(conv, 0), "corr_ci": [float(np.percentile(bs[0], 2.5)), float(np.percentile(bs[0], 97.5))],
                 "hist": stat(conv, 1), "hist_ci": [float(np.percentile(bs[1], 2.5)), float(np.percentile(bs[1], 97.5))]}
    out["locomo_cluster"] = lc
    # three-judge agreement over every judged answer (both benchmarks)
    keys = ["label_qwen3-8b", "label_qwen25-7b", "label_llama31"]
    vals = {k: [] for k in keys}
    for f in glob.glob(f"{ROOT}/results/main3/*/*.json") + glob.glob(f"{ROOT}/results/locomo/*/*.json"):
        for x in json.load(open(f))["answers"]:
            if all(k in x for k in keys):
                for k in keys:
                    vals[k].append(float(x[k]))
    out["judge_agreement"] = {f"{keys[i]}~{keys[j]}": {"kappa": kappa(vals[keys[i]], vals[keys[j]]),
                                                       "agree": float(np.mean(np.array(vals[keys[i]]) == np.array(vals[keys[j]])))}
                              for i in range(3) for j in range(i + 1, 3)}
    out["judge_n"] = len(vals[keys[0]])
    # memory size (storage) of the shared CoCoMem memory vs raw history
    st = [json.load(open(f)) for f in glob.glob(f"{ROOT}/states/cocomem/*.json")[:300]]
    out["storage"] = {"claims_tok": float(np.mean([sum(len(c["text"]) for c in d["claims"].values()) / 4 for d in st])),
                      "units_tok": float(np.mean([sum(len(u["text"]) for u in d["units"].values()) / 4 for d in st])),
                      "n_claims": float(np.mean([len(d["claims"]) for d in st])),
                      "n_units": float(np.mean([len(d["units"]) for d in st]))}
    json.dump(out, open(f"{ROOT}/results/{os.environ.get('X_OUT', 'extras6.json')}", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
