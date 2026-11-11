"""v22 write path: record-first compression of the derived units of the existing memories.

The claim ledger (L0), the unit structure (ids, anchored input lists, dates) and the clustering of the v21 memories
are kept; every note and digest is written again by the compressor in plan mode (CM_COMPRESSOR=plan, prompts.PLAN_*):
for every sentence the compressor first writes the derivation record the sentence is stated from and then the
sentence.  The records are therefore not a second pass over finished text but the plan the text was rendered from.
The rebuilt memories are the v22 states (CM_RUN_ROOT/states/{cocomem,locomo_cocomem}); the v21 states, the
counterfactual corrections and the evaluation are unchanged, so the two write paths are compared on identical claims
and identical unit structure.  The record-first memory has two derived levels: the per-owner profile of the v21
memories is not carried over (cocomem.PROFILE_LEVEL).

Servers on shared GPUs may run with a reduced context window (CM_MAXLEN): a unit whose prompt does not fit is left
pending in a partial state file and completed by a later pass on a server with the full window.

  python rebuild_units.py run   --kind lme    --units pairs         # the units of the labelled pairs first
  python rebuild_units.py run   --kind lme    --shard 0:3 --units rest             # the other units
  python rebuild_units.py run   --kind lme    --pending-only        # last pass: whatever is still pending (full window)
  python rebuild_units.py run   --kind locomo
  python rebuild_units.py count --kind lme
  python rebuild_units.py strip --kind lme                          # drop profile units left by earlier passes
  python rebuild_units.py stats                                     # write-time record statistics v21 vs v22"""
import argparse
import asyncio
import fcntl
import glob
import json
import os
import time

os.environ.setdefault("CM_COMPRESSOR", "plan")

import cocomem
from cocomem import CoCoMem, cited_ids
from llm import LLM, Usage, gather_limited, ContextOverflow

SRC = "/data/yaokang/corrmem"
ROOT = os.environ.get("CM_RUN_ROOT", f"{SRC}/v22")
SDIR = {"lme": f"{SRC}/states/cocomem", "locomo": f"{SRC}/states/locomo_cocomem"}
ODIR = {"lme": f"{ROOT}/states/cocomem", "locomo": f"{ROOT}/states/locomo_cocomem"}
PAIRS = [f"{SRC}/models/w20/train.jsonl", f"{SRC}/models/w20/test.jsonl"]


def qids_of(kind):
    """Memories the experiments touch: LongMemEval test200 and the users of the training / held-out pairs; LoCoMo all."""
    if kind == "locomo":
        return sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{SDIR['locomo']}/*.json"))
    cache = f"{ROOT}/qids_lme.json"                 # the pair files are large and the disk is slow: computed once
    if os.path.exists(cache):
        return json.load(open(cache))
    from splits import add_test200, make_splits
    q = set(add_test200(make_splits())["test200"])
    for p in PAIRS:
        with open(p) as stream:
            for line in stream:
                q.add(json.loads(line)["qid"])
    out = sorted(x for x in q if os.path.exists(f"{SDIR['lme']}/{x}.json"))
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def _add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def profile_units(units):
    return [x for x, u in units.items() if u["level"] == "profile"]


def cmd_strip(a):
    """Remove the profile units from the record-first states written so far (the memory has two derived levels) and
    from their pending lists.  Idempotent; every state is rewritten under its lock."""
    n = k = 0
    for q in qids_of(a.kind):
        out = f"{ODIR[a.kind]}/{q}.json"
        if not os.path.exists(out):
            continue
        with open(out + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = json.load(open(out))
            gone = profile_units(d["units"])
            if not gone:
                continue
            for x in gone:
                del d["units"][x]
            left = [x for x in (d.get("pending") or []) if x in d["units"]]
            d.pop("pending", None)
            if left:
                d["pending"] = left
            json.dump(d, open(out + ".tmp", "w"))
            os.replace(out + ".tmp", out)
            n += 1
            k += len(gone)
    print(f"strip {a.kind}: {k} profile units removed from {n} states", flush=True)


_PAIR_UNITS = None


def pair_units():
    """{qid: [unit ids]} of the units the training / held-out pairs are drawn from.  They are written first
    (run --units pairs) so that the labelling of the pairs can start while the other units are still being written."""
    global _PAIR_UNITS
    if _PAIR_UNITS is None:
        cache = f"{ROOT}/pair_units.json"
        if os.path.exists(cache):
            _PAIR_UNITS = json.load(open(cache))
        else:
            pu = {}
            for p in PAIRS:
                with open(p) as stream:
                    for line in stream:
                        r = json.loads(line)
                        pu.setdefault(r["qid"], set()).add(r["uid"])
            _PAIR_UNITS = {q: sorted(v) for q, v in pu.items()}
            os.makedirs(ROOT, exist_ok=True)
            json.dump(_PAIR_UNITS, open(cache, "w"))
    return _PAIR_UNITS


async def rebuild_one(llm, kind, qid, sem, pending_only, only=None, levels=None, rest=False):
    async with sem:
        return await _rebuild_one(llm, kind, qid, pending_only, only, levels, rest)


async def _rebuild_one(llm, kind, qid, pending_only, only=None, levels=None, rest=False):
    # the output is checked when the memory's turn comes (not when the coroutine is created), so that two processes
    # walking the same shard from opposite ends (--reverse) skip what the other has already written
    out = f"{ODIR[kind]}/{qid}.json"
    prev = json.load(open(out)) if os.path.exists(out) else None
    if prev is not None and not prev.get("pending"):
        return "cached"
    if prev is None and pending_only:
        return "skipped"
    d = prev if prev is not None else json.load(open(f"{SDIR[kind]}/{qid}.json"))
    m = CoCoMem.from_json(llm, d)
    for x in profile_units(m.units):            # the record-first memory has two derived levels (cocomem.PROFILE_LEVEL)
        del m.units[x]
    todo = [x for x in (prev["pending"] if prev is not None else m.units) if x in m.units]
    maxlen = int(os.environ.get("CM_MAXLEN", "0") or 0)
    usage, deferred = Usage(), []
    if only is not None or levels:          # this pass writes a subset of the units; the others stay pending
        keep = set(m.units) if only is None else set(only.get(qid, []))
        if rest:                            # the complement: every unit that is not a pair unit
            keep = set(m.units) - keep
        if levels:
            keep = {x for x in keep if x in m.units and m.units[x]["level"] in levels}
        deferred = [x for x in todo if x not in keep]
        todo = [x for x in todo if x in keep]
        if prev is not None and not todo:
            return "cached"
    t0 = time.time()

    async def one(uid):
        u = m.units[uid]
        cids = [c for c in u["inputs"] if c in m.claims]
        if not cids:
            return
        prompt = m._unit_prompt_for(u["level"], cids, date=u.get("date"), person=u.get("person"))
        if maxlen and maxlen < cocomem.FULL_WINDOW and len(prompt) // 3 + CoCoMem.PLAN_TOKENS[u["level"]] + 64 > maxlen:
            deferred.append(uid)                 # a reduced window: left to a pass on the full window
            return
        try:
            w = await m._compress(u["level"], cids, usage, date=u.get("date"), person=u.get("person"))
        except ContextOverflow:
            deferred.append(uid)
            return
        m._set_compressed(u, w)
        u["written"] = "plan"

    await asyncio.gather(*(one(x) for x in todo))
    done = [x for x in todo if x not in set(deferred)]
    os.makedirs(ODIR[kind], exist_ok=True)
    # several passes may work on one memory at the same time (pair units, profiles, the other units): the units this
    # pass has written are merged into the state as it is now, under a lock, so that no pass undoes another
    with open(out + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.load(open(out)) if os.path.exists(out) else None
        if cur is None:
            d2 = m.to_json()
            d2["build_time"] = d.get("build_time")
            d2["plan_usage"], d2["plan_time"] = usage.as_dict(), time.time() - t0
            left = sorted(deferred)
        else:
            d2 = cur
            for x in done:
                d2["units"][x] = m.units[x]
            d2["compressor"] = cocomem.COMPRESSOR
            d2["plan_usage"] = _add(cur.get("plan_usage", {}), usage.as_dict())
            d2["plan_time"] = cur.get("plan_time", 0) + time.time() - t0
            left = sorted(set(cur.get("pending") or []) - set(done))
        for x in profile_units(d2["units"]):
            del d2["units"][x]
        left = [x for x in left if x in d2["units"]]
        d2.pop("pending", None)
        if left:
            d2["pending"] = left
        json.dump(d2, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
    return "pending" if left else "ok"


async def cmd_run(a):
    assert cocomem.COMPRESSOR == "plan", "CM_COMPRESSOR must be plan"
    qids = qids_of(a.kind)
    if a.shard:
        i, n = (int(x) for x in a.shard.split(":"))
        qids = qids[i::n]
    if a.limit:
        qids = qids[:a.limit]
    if a.reverse:
        qids = qids[::-1]
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=a.conc)
    sem = asyncio.Semaphore(a.inst)
    only = pair_units() if a.units in ("pairs", "rest") else None
    levels = set(a.levels.split(",")) if a.levels else None
    res = await gather_limited([rebuild_one(llm, a.kind, q, sem, a.pending_only, only, levels, a.units == "rest") for q in qids],
                               desc=f"rebuild-{a.kind}", every=5)
    print({r: res.count(r) for r in set(res)}, flush=True)


def cmd_count(a):
    """<memories> <complete> <pending memories> <pending units> <pair units still pending>  (for the job scripts)."""
    qids = qids_of(a.kind)
    pu_of = pair_units() if a.kind == "lme" else {}
    n = done = pend = pu = pp = 0
    for q in qids:
        n += 1
        p = f"{ODIR[a.kind]}/{q}.json"
        if not os.path.exists(p):
            pp += len(pu_of.get(q, []))
            continue
        d = json.load(open(p))
        if d.get("pending"):
            pend += 1
            pu += len(d["pending"])
            pp += len(set(d["pending"]) & set(pu_of.get(q, [])))
        else:
            done += 1
    print(n, done, pend, pu, pp)


def _unit_stats(u, m):
    """Per-unit record statistics.  sents / with_rec: sentences, sentences with a record.  inputs / covered: anchored
    inputs, inputs named by at least one record or citation of the unit.  rec_ids / cite_ids: facts named by the
    records / cited by the sentences (summed over sentences).  listing: records that name at least 80% of the unit's
    inputs, counted in units of six or more inputs (sents_big sentences): a record that restates the input list
    instead of the sentence's own derivation.  rec_tok: record tokens; words: unit text length."""
    sents = u.get("sents") or []
    recs = u.get("deriv") or []
    ins = [c for c in u["inputs"] if c in m.claims]
    S, named, nid, ncite, lst = set(ins), set(), 0, 0, 0
    big = len(ins) >= 6
    for r, s in zip(recs, sents):
        rid, cid = set(cocomem._RID.findall(r or "")) & S, set(cited_ids(s)) & S
        named |= rid | cid
        nid += len(rid)
        ncite += len(cid)
        lst += int(big and len(rid) >= 0.8 * len(ins))
    live = [r for r in recs if r and r.strip().lower() != "none"]
    from commutator21 import is_keyed, record_entries
    ent = [record_entries(r) for r in live]
    return {"sents": len(sents), "with_rec": len(live), "inputs": len(ins), "covered": len(named),
            "rec_tok": sum(cocomem.ntok(r) for r in live), "words": len(u["text"].split()),
            "rec_ids": nid, "cite_ids": ncite, "listing": lst, "sents_big": len(sents) if big else 0,
            # keyed: every claim the record names has its own operand entry; multi: records with two or more operands
            "keyed": sum(is_keyed(r) for r in live), "multi": sum(len({k for k, _ in e if k}) >= 2 for e in ent),
            "entries": sum(len(e) for e in ent), "operand_entries": sum(k is not None for e in ent for k, _ in e)}


def cmd_stats(a):
    """Write-time record statistics of the two write paths on the same claims and unit structure.
      v21   records attributed to the finished prose by a second pass (states/deriv), on the units that pass covered
      v22p  record-first, on exactly those units (the paired comparison)
      v22   record-first, on all units of the memories
    Coverage = share of a unit's anchored inputs that at least one record (or citation) of the unit names: the share
    of corrections for which the sentence closure S(c) is defined.  plan_ok = units written through the grammar."""
    from deriv import attach as attach_deriv
    out = {}

    def add(agg, lev, s, ok=None):
        g = agg.setdefault(lev, {})
        for k, v in s.items():
            g[k] = g.get(k, 0) + v
        g["units"] = g.get("units", 0) + 1
        if ok is not None:
            g["plan_ok"] = g.get("plan_ok", 0) + int(ok)

    for kind in ("lme", "locomo"):
        ddir = f"{SRC}/states/deriv" if kind == "lme" else f"{SRC}/states/deriv_locomo"
        agg = {"v21": {}, "v22p": {}, "v22": {}}
        n_mem = 0
        for q in qids_of(kind):
            p22 = f"{ODIR[kind]}/{q}.json"
            if not os.path.exists(p22):
                continue
            d22 = json.load(open(p22))
            if d22.get("pending") and not a.partial:
                continue
            pend = set(d22.get("pending") or [])
            m21 = CoCoMem.from_json(None, json.load(open(f"{SDIR[kind]}/{q}.json")))
            m22 = CoCoMem.from_json(None, d22)
            attach_deriv(m21, f"{ddir}/{q}.json")
            n_mem += 1
            for uid, u22 in m22.units.items():
                if uid in pend or u22.get("written") != "plan":
                    continue
                u21 = m21.units.get(uid)
                s22 = _unit_stats(u22, m22)
                add(agg["v22"], u22["level"], s22, bool(u22.get("plan_ok")))
                if u21 is not None and "deriv" in u21:           # same unit, same input list
                    add(agg["v21"], u22["level"], _unit_stats(u21, m21))
                    add(agg["v22p"], u22["level"], s22, bool(u22.get("plan_ok")))
        for ver in agg:
            tot = {}
            for lev, s in list(agg[ver].items()):
                for k, v in s.items():
                    tot[k] = tot.get(k, 0) + v
            agg[ver]["all"] = tot
            for lev, s in agg[ver].items():
                if not s:
                    continue
                s["coverage"] = 100 * s["covered"] / max(1, s["inputs"])
                s["rec_rate"] = 100 * s["with_rec"] / max(1, s["sents"])
                s["words_per_unit"] = s["words"] / max(1, s["units"])
                s["sents_per_unit"] = s["sents"] / max(1, s["units"])
                s["rec_tok_per_sent"] = s["rec_tok"] / max(1, s["sents"])
                s["ids_per_rec"] = s["rec_ids"] / max(1, s["sents"])
                s["cites_per_sent"] = s["cite_ids"] / max(1, s["sents"])
                s["listing_rate"] = 100 * s["listing"] / max(1, s["sents_big"])
                s["keyed_rate"] = 100 * s["keyed"] / max(1, s["with_rec"])
                s["multi_rate"] = 100 * s["multi"] / max(1, s["with_rec"])
                s["operand_entry_rate"] = 100 * s["operand_entries"] / max(1, s["entries"])
                if "plan_ok" in s:
                    s["plan_ok_rate"] = 100 * s["plan_ok"] / max(1, s["units"])
        out[kind] = {"memories": n_mem, **agg}
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    name = "write_stats_v22_partial.json" if a.partial else "write_stats_v22.json"
    json.dump(out, open(f"{ROOT}/results/{name}", "w"), indent=1)
    for kind in out:
        for ver in ("v21", "v22p", "v22"):
            for lev, s in out[kind][ver].items():
                if s:
                    print(f"{kind:6s} {ver:4s} {lev:7s} units={s['units']:6d} sents/unit={s['sents_per_unit']:.1f} words/unit={s['words_per_unit']:.0f} "
                          f"rec%={s['rec_rate']:.1f} ids/rec={s['ids_per_rec']:.2f} cites/sent={s['cites_per_sent']:.2f} "
                          f"listing%={s['listing_rate']:.1f} cov%={s['coverage']:.1f} rec tok/sent={s['rec_tok_per_sent']:.0f} "
                          f"grammar%={s.get('plan_ok_rate', float('nan')):.1f} keyed%={s['keyed_rate']:.2f} multi-operand%={s['multi_rate']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "count", "stats", "strip"])
    ap.add_argument("--kind", default="lme", choices=["lme", "locomo"])
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--inst", type=int, default=4)
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="walk the shard backwards (helper for a slow shard)")
    ap.add_argument("--units", default="all", choices=["all", "pairs", "rest"],
                    help="pairs: only the units of the training / held-out pairs; rest: every other unit")
    ap.add_argument("--levels", default="", help="e.g. note,digest: only the units of these levels (the others stay pending)")
    ap.add_argument("--partial", action="store_true", help="stats over the units written so far (progress check)")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(cmd_run(a))
    elif a.cmd == "count":
        cmd_count(a)
    elif a.cmd == "strip":
        cmd_strip(a)
    else:
        cmd_stats(a)
