"""Deletion adversaries for the direct test (reviewer: "a method that deletes derived conclusions it cannot repair lowers
staleness without repairing anything").  Two strategies are built from existing runs, without any LLM:
  del_oracle  delete exactly the sentences that recompression changed (the best possible deleter: it needs the answer)
  del_comm10  delete the sentences that CoCoMem rewrites, instead of rewriting them
The result files copy `recompress` and replace the repaired units of scenario S1, so comm_eval2 judges them like any strategy."""
import glob
import json
import os
import re
import sys

from cocomem import split_sentences

ROOT = "/data/yaokang/corrmem/results/main3"
sents = lambda t: [re.sub(r"\s+", " ", t[a:b]).strip() for a, b in split_sentences(t)]
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())


def build(name, other):
    out = f"{ROOT}/{name}"
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{ROOT}/recompress/*.json")):
        d = json.load(open(f))
        g = f"{ROOT}/{other}/{os.path.basename(f)}"
        if not os.path.exists(g):
            continue
        if "S1" not in d.get("units", {}) or "S0" not in d["units"]:
            continue
        ref = d["units"]["S1"]
        base = d["units"]["S0"]
        gd = json.load(open(g))
        if "S1" not in gd.get("units", {}):
            continue
        src = gd["units"]["S1"] if other != "recompress" else ref
        new = dict(base)
        for uid in d["repair"]["S1"].get("closure", []):
            if uid not in base or uid not in src:
                continue
            changed = {norm(s) for s in sents(base[uid])} - {norm(s) for s in sents(src[uid])}
            new[uid] = " ".join(s for s in sents(base[uid]) if norm(s) not in changed)
        d["units"] = {**d["units"], "S1": new}
        d["strategy"] = name
        json.dump(d, open(f"{out}/{os.path.basename(f)}", "w"))
        n += 1
    print(name, n)


if __name__ == "__main__":
    build("del_oracle", "recompress")
    build("del_comm10", "comm10")
