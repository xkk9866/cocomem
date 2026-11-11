"""Run correction scenarios (S0/S1/S2) for CoCoMem maintenance strategies."""
import argparse
import asyncio
import json
import os
import time

from cocomem import CoCoMem
from data import load_lme
from llm import LLM, Usage, gather_limited
from scenarios import corr_session, load_corr, queries
from splits import add_test200, make_splits

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
STATES = f"{ROOT}/states/cocomem"
RES = f"{ROOT}/results"
# also ask the main question of S0 and S1 with the derived views as the only context (answers with qkey "qv")
VIEWS_ONLY = os.environ.get("CM_VIEWS_ONLY", "1") != "0"

_CLF = None


def get_clf(path):
    global _CLF
    if _CLF is None:
        from ncf import NCF
        _CLF = NCF(path)
    return _CLF


_TNET = None


def get_tnet(name="tnet"):
    """Client of the distilled commutator, served (int4) on its own port (start_tnet.sh)."""
    global _TNET
    if _TNET is None or getattr(_TNET, "logical_model", _TNET.model) != name:
        _TNET = LLM(name, int(os.environ.get("CM_PT", 18432)), concurrency=256)
    return _TNET


def lineage_cached(base, qid, mode, args):
    """Field lineage is a write-time index: compute once per memory and cache it."""
    p = f"{ROOT}/states/lineage_{mode}_{args.clf_tag}/{qid}.json"
    if os.path.exists(p):
        d = json.load(open(p))
        for uid, v in d.items():
            if uid in base.units:
                base.units[uid]["lineage"][mode] = v["lineage"]
                base.units[uid]["support"] = v["support"]
        base.clf = get_clf(args.clf)
        return
    base.set_lineage(mode, cle=get_clf(args.clf), thr=args.thr)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({uid: {"lineage": u["lineage"][mode], "support": u.get("support", {})} for uid, u in base.units.items()},
              open(p, "w"))


async def run_one(llm, inst, rec, strategy, args, sem):
    out_path = f"{RES}/{args.tag}/{strategy}/{inst['qid']}.json"
    if os.path.exists(out_path):
        return "cached"
    st_path = f"{STATES}/{inst['qid']}.json"
    if not os.path.exists(st_path):
        return "nostate"
    async with sem:
        base = CoCoMem.from_json(llm, json.load(open(st_path)))
        from cocomem import STRATEGIES
        lin = STRATEGIES.get(strategy, {}).get("lineage")
        if lin == "ncf":
            lineage_cached(base, inst["qid"], "ncf", args)
        elif lin:
            base.set_lineage(lin)
        if STRATEGIES.get(strategy, {}).get("repair") == "ccf":
            base.clf = get_clf(args.clf)
        if STRATEGIES.get(strategy, {}).get("tnet") == "student":
            base.tnet = get_tnet(STRATEGIES[strategy].get("tnet_name", "tnet"))
        if STRATEGIES.get(strategy, {}).get("witness") == "annot":     # write-time derivation witness (witness.py)
            from witness import attach
            attach(base, f"{ROOT}/states/witness/{inst['qid']}.json")
        if STRATEGIES.get(strategy, {}).get("deriv"):                   # write-time derivation records (deriv.py)
            from deriv import attach as attach_deriv
            attach_deriv(base, f"{ROOT}/states/deriv/{inst['qid']}.json")
        res = {"qid": inst["qid"], "qtype": inst["qtype"], "strategy": strategy, "answers": [], "repair": {}}
        mems = {"S0": base}

        async def corr(kind, sc):
            m = base.clone()
            u = Usage()
            t0 = time.time()
            info = await m.correct(corr_session(inst, rec, kind), strategy=strategy, usage=u)
            info["usage"] = u.as_dict()
            info["time"] = time.time() - t0
            res["repair"][sc] = info
            res.setdefault("units", {})[sc] = {uid: x["text"] for uid, x in m.units.items()}
            mems[sc] = m

        await asyncio.gather(*(corr(kind, sc) for kind, sc in (("affect", "S1"), ("unrel", "S2")) if kind in rec))
        from cocomem import reader_answer

        async def ask(sc, qid, q, gold, jt):
            u = Usage()
            # "qv": the same question read from the derived views alone (notes and digests, no ledger entries)
            _, ctx = await mems[sc].retrieve(q, budget=args.budget, levels=("note", "digest") if qid == "qv" else None)
            hyp, full = await reader_answer(llm, ctx, q, inst["qdate"], u)
            stale = None
            if sc == "S1" and qid == "q":
                old = rec["affect"].get("history_answer", "")
                stale = bool(old) and old.lower() in ctx.lower()
            return {"scenario": sc, "qkey": qid, "question": q, "gold": gold, "jtype": jt,
                    "hyp": hyp, "full": full, "ctx_tokens": len(ctx) // 4, "stale_in_ctx": stale}

        qs = queries(inst, rec)
        if VIEWS_ONLY:
            qs += [(sc, "qv", q, gold, jt) for sc, qid, q, gold, jt in qs if qid == "q" and sc in ("S0", "S1")]
        res["answers"] = list(await asyncio.gather(*(ask(*x) for x in qs)))
        res.setdefault("units", {})["S0"] = {uid: x["text"] for uid, x in base.units.items()}
        res["mem_size"] = base.size_tokens()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path, "w"))
        return "ok"


async def main(args):
    sp = add_test200(make_splits())
    qids = []
    for s in args.splits.split(","):
        qids += sp[s]
    if args.limit:
        qids = qids[:args.limit]
    if os.environ.get("CM_REVERSE"):
        qids = qids[::-1]
    items = {x["qid"]: x for x in load_lme()}
    corr = load_corr()
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=args.conc)
    sem = asyncio.Semaphore(args.inst)
    for strategy in args.strategies.split(","):
        res = await gather_limited([run_one(llm, items[q], corr[q], strategy, args, sem) for q in qids if q in corr],
                                   desc=strategy, every=10)
        print(strategy, {r: res.count(r) for r in set(res)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--strategies", default="append,coarse")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--clf", default="/data/yaokang/corrmem/models/ncf")
    ap.add_argument("--clf_tag", default="v1")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--inst", type=int, default=16)
    asyncio.run(main(ap.parse_args()))
