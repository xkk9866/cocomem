"""Held-out sentence--correction pairs with commutation-verified targets: on the sentences a counterfactual correction
changes, does the record name the corrected claim, and do the inline citations?  By level.  Also the other direction:
of the sentences whose record names the corrected claim, the share the correction changes.
  python rec_recall22.py name=labels.jsonl [name=labels.jsonl ...] out.json        (CPU, no model)"""
import json
import re
import sys
from collections import Counter


def cited(s):
    return {c for g in re.findall(r"\[([^\]]*)\]", s) for c in re.findall(r"c\d+", g)}


def main():
    out = {}
    for spec in sys.argv[1:-1]:
        name, path = spec.split("=", 1)
        try:
            E = [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            continue
        C = Counter()
        for e in E:
            m = re.match(r"\[(c\d+)\]", e["old"])
            if not m:
                continue
            c, lev = m.group(1), e["level"]
            ids = set(e.get("Dids") or [])
            rec, cit = c in ids, c in cited(e["sent"])
            for L in (lev, "all"):
                C[L, "pairs"] += 1
                C[L, "named"] += rec
                C[L, "named_changed"] += rec and bool(e["y"])
                if e["y"]:
                    C[L, "n"] += 1
                    C[L, "rec"] += rec
                    C[L, "cit"] += cit
                    C[L, "either"] += rec or cit
                    C[L, "fanin"] += len(ids)
        r = {}
        for L in sorted({k[0] for k in C}):
            n = C[L, "n"]
            r[L] = {"pairs": C[L, "pairs"], "n": n, "rec": round(100 * C[L, "rec"] / max(1, n), 1),
                    "cit": round(100 * C[L, "cit"] / max(1, n), 1), "either": round(100 * C[L, "either"] / max(1, n), 1),
                    "fanin": round(C[L, "fanin"] / max(1, n), 2), "named": C[L, "named"],
                    "named_changed": round(100 * C[L, "named_changed"] / max(1, C[L, "named"]), 1)}
            print(name, L, json.dumps(r[L]))
        out[name] = r
    json.dump(out, open(sys.argv[-1], "w"), indent=1)


if __name__ == "__main__":
    main()
