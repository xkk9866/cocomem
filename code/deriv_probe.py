"""Probe the record pass: operand recall on the synthetic units (program truth) and the entry format on real units.

  CM_DERIV_PROMPT=v2 CM_P8=18550 python deriv_probe.py --synthetic      # recall/complete of the records of 150 units
  python deriv_probe.py --real                                           # format statistics of states/deriv records"""
import argparse
import asyncio
import glob
import json
import os
import re

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
_ENTRY = re.compile(r"^\s*\[?c\d+\]?\s*=")


def fmt_stats(records):
    n = tot = good = none = noid = multi = 0
    for r in records:
        n += 1
        if r.strip().lower() == "none":
            none += 1
            continue
        parts = [p.strip() for p in r.split(";") if p.strip()]
        ids = re.findall(r"\bc\d+\b", r)
        if not ids:
            noid += 1
        for p in parts:
            tot += 1
            if _ENTRY.match(p):
                good += 1
            elif re.match(r"^\s*\[?c\d+\s*,", p):
                multi += 1
    return {"records": n, "none": none, "no_id": noid, "entries": tot, "id=value entries": good, "multi-id entries": multi,
            "good_share": round(100 * good / max(1, tot), 1)}


async def synthetic(a):
    from deriv import DERIV
    from llm import LLM, Usage
    from suffbench import build
    from suffbench5 import KINDS, parse_records
    cases, units = build(150, 0)
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=a.conc)
    u = Usage()

    async def one(unit):
        sents = [unit["S"][k] for k in KINDS]
        facts = "\n".join(f"[c{j + 1}] ({unit['days'][j][:3]}, user) {f}" for j, f in enumerate(unit["facts"]))
        shown = "\n".join(f"({k}) {s}" for k, s in enumerate(sents))
        r = await llm.chat(DERIV.format(kind="digest", facts=facts, unit=shown), max_tokens=900, usage=u)
        return dict(zip(KINDS, parse_records(r, len(sents))))

    Ds = await asyncio.gather(*(one(x) for x in units))
    rec = [len(set(re.findall(r"\bc\d+\b", Ds[j][k])) & {f"c{i + 1}" for i in range(len(x["vals"]))}) / len(x["vals"])
           for j, x in enumerate(units) for k in KINDS[:-1]]
    out = {"prompt": os.environ.get("CM_DERIV_PROMPT", "v1"), "recall": 100 * sum(rec) / len(rec),
           "complete": 100 * sum(r == 1 for r in rec) / len(rec), "tokens_per_unit": (u.prompt_tokens + u.completion_tokens) / len(units),
           "format": fmt_stats([Ds[j][k] for j in range(len(units)) for k in KINDS[:-1]])}
    print(json.dumps(out, indent=1))
    for j in range(3):
        for k in KINDS[:3]:
            print(k, "|", Ds[j][k][:200])


def real(a):
    recs = {"note": [], "digest": [], "profile": []}
    for f in sorted(glob.glob(f"{ROOT}/states/deriv/*.json"))[:120]:
        d = json.load(open(f))
        for uid, v in d["units"].items():
            lev = "profile" if uid.startswith("profile") else "digest" if uid.startswith("d") else "note"
            recs[lev] += v["deriv"]
    for lev, rs in recs.items():
        print(lev, json.dumps(fmt_stats(rs)))
    import random
    rnd = random.Random(0)
    for lev in ("digest", "profile"):
        for r in rnd.sample(recs[lev], 6):
            print(lev, "|", r[:220])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    if a.real:
        real(a)
    if a.synthetic:
        asyncio.run(synthetic(a))
