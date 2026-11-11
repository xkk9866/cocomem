"""End-to-end maintenance latency without the LLM cache (CM_NOCACHE=1 must be set).

Every strategy repairs the same sample of evidence corrections, one correction at a time (single-request
latency, as an agent would experience it), on the same GPU and servers.  We report wall-clock seconds,
8B tokens, commutator (4B) tokens and FLOP-weighted 8B-equivalent tokens (2 * parameters * tokens)."""
import argparse
import asyncio
import json
import os
import random
import time
from statistics import median, quantiles

assert os.environ.get("CM_NOCACHE"), "set CM_NOCACHE=1"
from cocomem import CoCoMem, STRATEGIES
from data import load_lme
from llm import LLM, Usage
from run_corr import STATES, get_tnet, lineage_cached
from scenarios import corr_session, load_corr
from splits import add_test200, make_splits

ROOT = "/data/yaokang/corrmem"
P8, P4 = 8.2, 4.0      # billions of parameters of the compressor and of the commutator


async def main(a):
    sp = add_test200(make_splits())
    corr = load_corr()
    items = {x["qid"]: x for x in load_lme()}
    qs = [q for q in sp["test200"] if q in corr and "affect" in corr[q] and os.path.exists(f"{STATES}/{q}.json")]
    qs = random.Random(0).sample(qs, a.n)
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=128)
    out = {}
    for strat in a.methods.split(","):
        rows = []

        async def one(q):
            m = CoCoMem.from_json(llm, json.load(open(f"{STATES}/{q}.json")))
            cfg = STRATEGIES.get(strat, {})
            if cfg.get("lineage") == "ncf":
                lineage_cached(m, q, "ncf", argparse.Namespace(clf_tag="v2", clf=f"{ROOT}/models/ccf", thr=0.5))
            elif cfg.get("lineage"):
                m.set_lineage(cfg["lineage"])
            if cfg.get("tnet") == "student":
                m.tnet = get_tnet(cfg.get("tnet_name", "tnet"))
            if cfg.get("witness") == "annot":
                from witness import attach
                attach(m, f"{ROOT}/states/witness/{q}.json")
            if cfg.get("deriv"):
                from deriv import attach as attach_deriv
                attach_deriv(m, f"{ROOT}/states/deriv/{q}.json")
            u = Usage()
            t0 = time.time()
            info = await m.correct(corr_session(items[q], corr[q], "affect"), strategy=strat, usage=u)
            dt = time.time() - t0
            t8 = u.prompt_tokens + u.completion_tokens
            t4 = (info.get("t_usage") or {}).get("in", 0) + (info.get("t_usage") or {}).get("out", 0)
            return {"q": q, "sec": dt, "tok8": t8, "tok4": t4, "eq8": t8 + t4 * P4 / P8,
                    "ledger": sum((info.get("ledger_sents") or {}).values())}

        wall0 = time.time()
        for i in range(0, len(qs), a.par):
            rows += await asyncio.gather(*(one(q) for q in qs[i:i + a.par]))
        wall = time.time() - wall0
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        srt = sorted(r["sec"] for r in rows)
        out[strat] = {"n": len(rows), "sec": mean("sec"), "p50": median(srt), "p90": quantiles(srt, n=10, method="inclusive")[8] if len(srt)>1 else srt[0],
                      "tok8": mean("tok8"), "tok4": mean("tok4"), "eq8": mean("eq8"), "ledger": mean("ledger"),
                      "par": a.par, "wall": wall, "per_min": 60 * len(rows) / wall}
        print(strat, json.dumps(out[strat]), flush=True)
    json.dump(out, open(f"{ROOT}/results/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="latency.json")
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--methods", default="comm3,comm2,ncf,coarse,recompress,spec,rebuild,simupdate,slots")
    asyncio.run(main(ap.parse_args()))
