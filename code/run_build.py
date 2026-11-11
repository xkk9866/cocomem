"""Build CoCoMem memories (write path) for a split of LongMemEval-S."""
import argparse
import asyncio
import json
import os
import time

from cocomem import CoCoMem
from data import load_lme
from llm import LLM, gather_limited
from splits import add_test200, make_splits

OUT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem") + "/states/cocomem"


async def build_one(llm, inst, sem):
    path = f"{OUT}/{inst['qid']}.json"
    if os.path.exists(path):
        return "cached"
    async with sem:
        m = CoCoMem(llm)
        t0 = time.time()
        # claim extraction of all sessions is independent -> run concurrently, then insert in order
        raws = await asyncio.gather(*(m.extract(s, m.usage["write"]) for s in inst["sessions"]))
        notes = []
        for s, raw in zip(inst["sessions"], raws):
            new = []
            for r in raw:
                cid = m._new_cid()
                m.claims[cid] = {"id": cid, "text": r["text"], "speaker": r["speaker"], "turn": r["turn"],
                                 "sid": s["sid"], "date": s["date"], "status": "current", "superseded_by": [],
                                 "superseded_at": None, "revises": [], "revises_hint": r["revises"]}
                new.append(cid)
            if new:
                notes.append(m._make_note(s, new, m.usage["write"]))
        await asyncio.gather(*notes)
        await m.consolidate()
        d = m.to_json()
        d["build_time"] = time.time() - t0
        os.makedirs(OUT, exist_ok=True)
        json.dump(d, open(path, "w"))
        return "ok"


async def main(args):
    sp = add_test200(make_splits())
    qids = []
    for s in args.splits.split(","):
        qids += sp[s]
    if args.qids:                                 # explicit list (e.g. the users of the counterfactual pair files)
        qids = [q for q in args.qids.split(",") if q]
    qids = list(dict.fromkeys(qids))
    if args.shard:                                # "i:n": every n-th memory (parallel servers)
        i, n = (int(x) for x in args.shard.split(":"))
        qids = qids[i::n]
    if args.limit:
        qids = qids[:args.limit]
    items = {x["qid"]: x for x in load_lme()}
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=args.conc)
    sem = asyncio.Semaphore(args.inst)
    res = await gather_limited([build_one(llm, items[q], sem) for q in qids], desc="build", every=5)
    print({r: res.count(r) for r in set(res)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--inst", type=int, default=8)
    ap.add_argument("--qids", default="")
    ap.add_argument("--shard", default="")
    asyncio.run(main(ap.parse_args()))
