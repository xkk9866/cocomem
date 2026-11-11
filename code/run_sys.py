"""Build + run correction scenarios for end-to-end baseline systems."""
import argparse
import asyncio
import json
import os
import time

from baselines import SYSTEMS
from cocomem import reader_answer
from data import load_lme
from llm import LLM, Usage, gather_limited, ContextOverflow
from scenarios import corr_session, load_corr, queries
from splits import add_test200, make_splits

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")


async def build_one(llm, sysname, inst, sem):
    path = f"{ROOT}/states/{sysname}/{inst['qid']}.json"
    if os.path.exists(path):
        return json.load(open(path))
    async with sem:
        m = SYSTEMS[sysname](llm)
        t0 = time.time()
        await m.build(inst["sessions"])
        d = m.to_json()
        d["build_time"] = time.time() - t0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(d, open(path, "w"))
        return d


async def run_one(llm, sysname, inst, rec, args, sem_b, sem):
    out = f"{ROOT}/results/{args.tag}/{sysname}/{inst['qid']}.json"
    if os.path.exists(out):
        return "cached"
    try:
        d = await build_one(llm, sysname, inst, sem_b)
    except ContextOverflow:
        return "overflow"
    if rec is None:
        return "built"
    try:
        async with sem:
            base = SYSTEMS[sysname].from_json(llm, d)
            mems = {"S0": base}
            res = {"qid": inst["qid"], "qtype": inst["qtype"], "strategy": sysname, "answers": [], "repair": {}}
            for kind, sc in (("affect", "S1"), ("unrel", "S2")):
                if kind not in rec:
                    continue
                m = base.clone()
                u = Usage()
                t0 = time.time()
                await m.correct(corr_session(inst, rec, kind), usage=u)
                res["repair"][sc] = {"usage": u.as_dict(), "time": time.time() - t0}
                mems[sc] = m
            for sc, qk, q, gold, jt in queries(inst, rec):
                u = Usage()
                _, ctx = await mems[sc].retrieve(q, budget=args.budget)
                hyp, full = await reader_answer(llm, ctx, q, inst["qdate"], u, think=(sysname == "oracle"))
                stale = None
                if sc == "S1" and qk == "q":
                    old = rec["affect"].get("history_answer", "")
                    stale = bool(old) and old.lower() in ctx.lower()
                res["answers"].append({"scenario": sc, "qkey": qk, "question": q, "gold": gold, "jtype": jt,
                                       "hyp": hyp, "full": full, "ctx_tokens": len(ctx) // 4, "stale_in_ctx": stale})
            res["mem_size"] = base.size_tokens()
            res["build_usage"] = d["usage"]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(res, open(out, "w"))
            return "ok"
    except ContextOverflow:
        return "overflow"


async def main(args):
    sp = add_test200(make_splits())
    qids = []
    for s in args.splits.split(","):
        qids += sp[s]
    if args.shard:
        i, n = (int(x) for x in args.shard.split(":"))
        qids = qids[i::n]
    if args.reverse:
        qids = qids[::-1]
    if args.limit:
        qids = qids[:args.limit]
    items = {x["qid"]: x for x in load_lme()}
    corr = load_corr() if os.path.exists(f"{ROOT}/data/lme_corr.json") and not args.build_only else {}
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=args.conc)
    sem_b = asyncio.Semaphore(args.inst)
    sem = asyncio.Semaphore(args.inst)
    for sysname in args.systems.split(","):
        res = await gather_limited([run_one(llm, sysname, items[q], corr.get(q), args, sem_b, sem) for q in qids],
                                   desc=sysname, every=10)
        print(sysname, {r: res.count(r) for r in set(res)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="rawrag")
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--inst", type=int, default=12)
    ap.add_argument("--build_only", action="store_true")
    ap.add_argument("--shard", default="", help="i:n, this process runs questions i, i+n, ...")
    ap.add_argument("--reverse", action="store_true", help="take the shard from its end (a second worker on an unfinished shard)")
    asyncio.run(main(ap.parse_args()))
