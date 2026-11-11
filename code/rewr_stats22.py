"""v22 (CPU): what the sentences that a correction rewrites look like in their records.
For every evidence correction of the test set (scenario S1) and every sentence of the closure whose text differs after
the repair, read the record the sentence was written from (the S0 state): how many claims it names, and whether it
ends in a computation over them (a record whose last segment is `none` restates the listed values and computes
nothing).  The same two shares over every stored sentence of the test memories give the base rate.
  python rewr_stats22.py comm22,comm22_nd,... [out.json]"""
import glob
import json
import os
import re
import sys

from cocomem import split_sentences

P = "/data/yaokang/corrmem"
ROOT = os.environ.get("CM_RUN_ROOT", f"{P}/v22")
RC = ROOT + "/results"
norm = lambda t: re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", re.sub(r"\s*\[[^\]]*\]", "", t))).strip()


def comp(rec):
    """The computation of a record: the segment after the operand entries, or None when the record computes nothing."""
    seg = [x.strip() for x in (rec or "").split(";") if x.strip()]
    tail = [x for x in seg if not re.match(r"^c\d+\s*=", x)]
    tail = [x for x in tail if x.lower() != "none"]
    return tail[-1] if tail else None


def main():
    methods = sys.argv[1].split(",")
    E = json.load(open(f"{RC}/pub_export_qwen25-7b.json"))["main3"]
    valid = {q for q, r in E["oracle"].items() if r["lab"].get("S1:q") is True}
    out = {}
    base = {"sents": 0, "comp": 0, "multi": 0}
    seen = set()
    for m in methods:
        files = sorted(glob.glob(f"{RC}/main3/{m}/*.json"))
        if not files:
            continue
        S = {"n": 0, "comp": 0, "single": 0, "multi_comp": 0, "corr": 0, "corr_comp": 0, "valid_corr": 0, "valid_corr_comp": 0, "unmatched": 0}
        for f in files:
            d = json.load(open(f))
            q = d["qid"]
            rep = (d.get("repair") or {}).get("S1")
            sf = f"{ROOT}/states/cocomem/{q}.json"
            if rep is None or not os.path.exists(sf):
                continue
            st = json.load(open(sf))
            if q not in seen:
                seen.add(q)
                for u in st["units"].values():
                    for i, s in enumerate(u.get("sents") or []):
                        rec = (u.get("deriv") or [None] * (i + 1))[i] if i < len(u.get("deriv") or []) else None
                        ids = (u.get("dids") or [[]] * (i + 1))[i] if i < len(u.get("dids") or []) else []
                        base["sents"] += 1
                        base["comp"] += comp(rec) is not None
                        base["multi"] += len(ids) > 1
            anyc, n0 = False, S["n"]
            for uid in rep.get("closure") or []:
                u = st["units"].get(uid)
                b, c = d["units"]["S0"].get(uid), d["units"]["S1"].get(uid)
                if not u or b is None or c is None or b == c:
                    continue
                after = norm(c)                                 # a stored sentence is unchanged if the repaired unit still contains it
                for i, s in enumerate(u.get("sents") or []):
                    if norm(s) in after:
                        continue
                    rec = u["deriv"][i] if i < len(u.get("deriv") or []) else None
                    ids = u["dids"][i] if i < len(u.get("dids") or []) else []
                    S["n"] += 1
                    k = comp(rec) is not None
                    S["comp"] += k
                    S["single"] += len(ids) <= 1
                    S["multi_comp"] += k and len(ids) > 1
                    anyc = anyc or k
            if S["n"] > n0:
                S["corr"] += 1
                S["corr_comp"] += anyc
                if q in valid:
                    S["valid_corr"] += 1
                    S["valid_corr_comp"] += anyc
        pct = lambda a, b: round(100.0 * a / b, 1) if b else None
        out[m] = dict(S, comp_pct=pct(S["comp"], S["n"]), single_pct=pct(S["single"], S["n"]),
                      corr_comp_pct=pct(S["corr_comp"], S["corr"]), valid_corr_comp_pct=pct(S["valid_corr_comp"], S["valid_corr"]))
        print(m, json.dumps(out[m]))
    out["base"] = dict(base, comp_pct=round(100.0 * base["comp"] / max(1, base["sents"]), 1),
                       multi_pct=round(100.0 * base["multi"] / max(1, base["sents"]), 1))
    print("base", json.dumps(out["base"]))
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else f"{RC}/rewr_stats_v22.json", "w"), indent=1)


if __name__ == "__main__":
    main()
