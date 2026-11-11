"""Per-question labels of the published systems and CoCoMem, for the main tables and paired statistics."""
import argparse
import glob
import json
import os

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem") + "/results"


def labels(d, key):
    return {f"{a['scenario']}:{a['qkey']}": bool(a[key]) for a in d["answers"] if key in a}


def usage_tok(u):
    return (u or {}).get("in", 0) + (u or {}).get("out", 0)


def main(a):
    out = {"key": a.key, "main3": {}, "locomo": {}}
    canon = {}
    for f in glob.glob(f"{ROOT}/main3/{a.canon}/*.json"):
        d = json.load(open(f))
        canon[d["qid"]] = labels(d, a.key)
    for m in a.methods.split(",") + ["oracle"]:
        R = {}
        for f in glob.glob(f"{ROOT}/main3/{m}/*.json"):
            d = json.load(open(f))
            lab = labels(d, a.key)
            if m in a.shared.split(",") and d["qid"] in canon and "S0:q" in canon[d["qid"]]:
                lab["S0:q"] = canon[d["qid"]]["S0:q"]
            reps = [r for sc, r in d.get("repair", {}).items() if r and "usage" in r]
            R[d["qid"]] = {"qtype": d["qtype"], "lab": lab, "tok": [usage_tok(r.get("usage")) for r in reps],
                           "ttok": [usage_tok(r.get("t_usage")) for r in reps], "mem": d.get("mem_size"),
                           "build": usage_tok((d.get("build_usage") or {}).get("write"))}
        out["main3"][m] = R
    for m in a.loc_methods.split(","):
        R = {}
        for f in glob.glob(f"{ROOT}/locomo/{m}/*.json"):
            d = json.load(open(f))
            st = d.get("repair", {}).get("S1", {})
            steps = st.get("steps", [])
            R[d["cid"]] = {"lab": labels(d, a.key), "tok": usage_tok(st.get("usage")), "n": len(steps) or 1,
                           "ttok": sum(usage_tok(s.get("t_usage")) for s in steps)}
        out["locomo"][m] = R
    json.dump(out, open(a.out, "w"))
    print({k: {m: len(v) for m, v in out[k].items()} for k in ("main3", "locomo")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="label_qwen25-7b")
    ap.add_argument("--methods", default="rawrag,ga,membank,readagent,mem0,amem,comm21")
    ap.add_argument("--loc_methods", default="rawrag,ga,membank,readagent,mem0,amem,comm21")
    ap.add_argument("--shared", default="comm21")
    ap.add_argument("--canon", default="recompress")
    ap.add_argument("--out", default=f"{ROOT}/pub_export.json")
    main(ap.parse_args())
