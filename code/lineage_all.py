"""Write-time indexing: run the Neural Commutator Field over every (unit, input claim) pair of every
stored memory and cache the lineage / repair spans (states/lineage_ncf_<tag>/<qid>.json)."""
import argparse
import glob
import json
import os
import time

from cocomem import CoCoMem
from ncf import NCF


def main(a):
    clf = NCF(a.model)
    files = sorted(glob.glob(f"/data/yaokang/corrmem/states/{a.states}/*.json"))
    t0, n_pairs = time.time(), 0
    for i, p in enumerate(files):
        qid = os.path.basename(p)[:-5]
        out = f"/data/yaokang/corrmem/states/lineage_ncf_{a.tag}/{qid}.json"
        if os.path.exists(out):
            continue
        m = CoCoMem.from_json(None, json.load(open(p)))
        m.set_lineage("ncf", cle=clf, thr=a.thr)
        n_pairs += sum(len(u["inputs"]) for u in m.units.values())
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({uid: {"lineage": u["lineage"]["ncf"], "support": u.get("support", {})} for uid, u in m.units.items()},
                  open(out, "w"))
        if i % 20 == 0:
            print(f"{i}/{len(files)} pairs {n_pairs} {time.time() - t0:.0f}s", flush=True)
    print("done", n_pairs, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/yaokang/corrmem/models/ncf")
    ap.add_argument("--states", default="cocomem")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--thr", type=float, default=0.5)
    main(ap.parse_args())
