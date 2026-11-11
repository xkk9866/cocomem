"""Stage B for the official baselines: the shared reader (the same prompt, model and 4000-token budget as every other method)
answers from the contexts that the official systems retrieved (official_run.py), and the result files get the format of
run_sys.py / locomo.py so that judge.py and analyze.py treat them like any other method.
python official_read.py --system mem0 --task lme"""
import argparse
import asyncio
import json
import os

from cocomem import reader_answer
from llm import LLM, Usage, gather_limited

ROOT = "/data/yaokang/corrmem"
OUT = f"{ROOT}/bl/out"


def ntok(t):
    return len(t) // 4 + 1


async def lme(a, llm):
    tasks = {t["qid"]: t for t in json.load(open(f"{ROOT}/bl/tasks/lme.json"))}
    name = "off_" + a.system
    d = f"{OUT}/{a.tag}/{a.system}"

    async def one(f):
        r = json.load(open(f"{d}/{f}"))
        t = tasks[r["qid"]]
        out = f"{ROOT}/results/{a.res}/{name}/{r['qid']}.json"
        if os.path.exists(out):
            return "cached"
        hist = t["history_answer"]

        async def ask(sc, key, q, gold, jt):
            ctx = r["ctx"].get(f"{sc}:{key}", "")
            hyp, full = await reader_answer(llm, ctx, q, t["qdate"], Usage())
            stale = bool(hist) and hist.lower() in ctx.lower() if (sc == "S1" and key == "q") else None
            return {"scenario": sc, "qkey": key, "question": q, "gold": gold, "jtype": jt, "hyp": hyp, "full": full,
                    "ctx_tokens": len(ctx) // 4, "stale_in_ctx": stale}
        ans = await asyncio.gather(*(ask(*q) for q in t["queries"]))
        res = {"qid": r["qid"], "qtype": t["qtype"], "strategy": name, "answers": list(ans),
               "repair": {sc: v for sc, v in r["repair"].items()}, "build_usage": r["build"]["usage"],
               "build_time": r["build"]["time"], "mem_size": {"items": sum(ntok(x) for x in r["items"]["S0"])},
               "items": r["items"], "errors": r.get("errors", [])}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"))
        return "ok"
    files = sorted(f for f in os.listdir(d) if f.endswith(".json") and f[:-5] in tasks)
    r = await gather_limited([one(f) for f in files], desc=name, every=20)
    print(name, {x: r.count(x) for x in set(r)})


async def locomo(a, llm):
    tasks = {t["cid"]: t for t in json.load(open(f"{ROOT}/bl/tasks/locomo.json"))}
    name = "off_" + a.system
    d = f"{OUT}/{a.tag}/{a.system}"

    async def one(f):
        r = json.load(open(f"{d}/{f}"))
        t = tasks[r["cid"]]
        persons = t["persons"]
        out = f"{ROOT}/results/locomo/{name}/{r['cid']}.json"
        outr = f"{ROOT}/results/locomo_reg/{name}/{r['cid']}.json"
        rep = {"S1": {"usage": r["repair"]["usage"], "time": r["repair"]["time"], "steps": r["steps"]}}
        Q = t["queries"]

        async def ask(sc, key, q, gold, jt, cat):
            hyp, full = await reader_answer(llm, r["ctx"].get(f"{sc}:{key}", ""), q, "", Usage(), persons=persons)
            return {"scenario": sc, "qkey": key, "question": q, "gold": gold, "jtype": jt, "cat": cat, "hyp": hyp, "full": full}
        if not os.path.exists(out):
            ans = await asyncio.gather(*(ask(*q) for q in Q))
            res = {"cid": r["cid"], "method": name, "repair": rep, "answers": list(ans), "mem_size": {"items": sum(ntok(x) for x in r["items"]["S0"])},
                   "items": r["items"], "errors": r.get("errors", []), "build_usage": r["build"]["usage"], "build_time": r["build"]["time"]}
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(res, open(out, "w"))
        if not os.path.exists(outr):
            qi = {q[1]: q for q in Q if q[0] == "S1"}
            jobs = []

            async def rans(k, ctx):
                step, key = k.split(":")
                step = int(step)
                _, _, q, gold, jt, cat = qi[key]
                hyp, full = await reader_answer(llm, ctx, q, "", Usage(), persons=persons)
                kind = "unaffected" if key.startswith("u") else ("fresh" if int(key[1:]) == step else "carried")
                return {"scenario": f"R{step}", "qkey": key, "question": q, "gold": gold, "jtype": jt, "cat": cat, "hyp": hyp,
                        "full": full, "step": step, "kind": kind}
            ans = await asyncio.gather(*(rans(k, c) for k, c in r["reg"].items()))
            res = {"cid": r["cid"], "method": name, "repair": rep, "answers": list(ans)}
            os.makedirs(os.path.dirname(outr), exist_ok=True)
            json.dump(res, open(outr, "w"))
        return "ok"
    files = sorted(f for f in os.listdir(d) if f.endswith(".json") and f[:-5] in tasks)
    r = await gather_limited([one(f) for f in files], desc=name, every=5)
    print(name, {x: r.count(x) for x in set(r)})


async def main(a):
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=96)
    await (lme if a.task == "lme" else locomo)(a, llm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--task", default="lme")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--res", default="main3")
    asyncio.run(main(ap.parse_args()))
