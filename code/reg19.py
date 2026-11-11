"""Recurrence test on LoCoMo-Corr (sequential corrections, answers re-asked after EVERY correction).
carried: corrections made earlier, asked again after later corrections; fresh: asked right after their own correction;
unaffected: a fixed panel of five questions whose evidence was never corrected.  Recurrence: a correction that was
answered correctly right after it was made and is answered wrongly after a later correction."""
import glob
import json
import sys

import numpy as np

ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem") + "/results/locomo_reg"
KEY = "label_qwen25-7b"
methods = sys.argv[1].split(",")
out = {}
rng = np.random.default_rng(0)
for m in methods:
    files = sorted(glob.glob(f"{ROOT}/{m}/*.json"))
    if not files:
        continue
    per = []
    for f in files:
        d = json.load(open(f))
        A = [a for a in d["answers"] if KEY in a]
        fresh = [a[KEY] for a in A if a["kind"] == "fresh"]
        carried = [a[KEY] for a in A if a["kind"] == "carried"]
        unaff = [a[KEY] for a in A if a["kind"] == "unaffected"]
        byq = {}
        for a in A:
            if a["kind"] in ("fresh", "carried"):
                byq.setdefault(a["qkey"], {})[a["step"]] = a[KEY]
        ok_fresh = [q for q, v in byq.items() if any(s == int(q[1:]) and v[s] for s in v)]
        rec = [any(not byq[q][s] for s in byq[q] if s > int(q[1:])) for q in ok_fresh]
        # unaffected panel: accuracy after the first correction versus after the last (drift)
        u_first = [a[KEY] for a in A if a["kind"] == "unaffected" and a["step"] == 0]
        u_last = [a[KEY] for a in A if a["kind"] == "unaffected" and a["step"] == max(x["step"] for x in A)]
        per.append({"fresh": fresh, "carried": carried, "unaff": unaff, "rec": rec, "u_first": u_first, "u_last": u_last})
    cat = lambda k: [x for p in per for x in p[k]]
    mean = lambda v: 100 * float(np.mean(v)) if len(v) else float("nan")

    def cb(k, B=2000):                    # bootstrap over conversations
        s = []
        for _ in range(B):
            idx = rng.integers(0, len(per), len(per))
            v = [x for i in idx for x in per[i][k]]
            s.append(100 * np.mean(v) if v else np.nan)
        return [float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))]
    out[m] = {"fresh": mean(cat("fresh")), "carried": mean(cat("carried")), "carried_ci": cb("carried"),
              "recurrence": mean(cat("rec")), "recurrence_ci": cb("rec"), "unaffected": mean(cat("unaff")),
              "unaffected_first": mean(cat("u_first")), "unaffected_last": mean(cat("u_last")),
              "n_conv": len(per), "n_carried": len(cat("carried"))}
json.dump(out, open(f"{ROOT}/../reg19.json", "w"), indent=1)
print(json.dumps(out, indent=1))
