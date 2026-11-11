"""Judge-free staleness check.  A unit of the closure is 'eligible' when it contained a number of the old claim that the
correction message does not repeat; it 'leaks' when the repaired unit still contains such a number.  Pure string test:
no model, no threshold.  Output: results/leak2.json  {method: {units, eligible, leak, leak_q}}."""
import json
import re
import sys

import numpy as np

from comm_eval3 import load, ROOT

NUM = re.compile(r"\d+(?:[.,]\d+)?")
nums = lambda s: {x.replace(",", "") for x in NUM.findall(s)}
tag, ref = "main3", "oracle"
methods = sys.argv[1].split(",")
corr = {r["qid"]: r for r in json.load(open(f"{ROOT}/data/lme_corr.json"))}
R0 = load(tag, "comm10")
out = {}
for m in methods:
    R = load(tag, m)
    rows = []
    for q, ref_d in R0.items():
        d = R.get(q)
        if not d or "S1" not in d.get("units", {}) or "S1" not in ref_d.get("repair", {}):
            continue
        rev, clo = ref_d["repair"]["S1"].get("revised", []), ref_d["repair"]["S1"].get("closure", [])
        if not rev or not clo:
            continue
        st = json.load(open(f"{ROOT}/states/cocomem/{q}.json"))
        old = " ".join(st["claims"][c]["text"] for c in rev if c in st["claims"])
        only = nums(old) - nums(corr[q]["affect"]["correction_message"])
        base, cand = d["units"]["S0"], d["units"]["S1"]
        for uid in clo:
            if uid in cand and uid in base and only & nums(base[uid]):
                rows.append((q, bool(only & nums(cand[uid]))))
    if not rows:
        continue
    byq = {}
    for q, l in rows:
        byq.setdefault(q, []).append(l)
    out[m] = {"eligible": len(rows), "leak": 100 * float(np.mean([l for _, l in rows])),
              "leak_q": 100 * float(np.mean([any(v) for v in byq.values()])), "questions": len(byq)}
    print(m, out[m], flush=True)
json.dump(out, open(f"{ROOT}/results/leak2.json", "w"), indent=1)
