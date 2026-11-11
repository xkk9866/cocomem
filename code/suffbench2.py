"""Programmatic benchmark v2: the single three-outcome commutator, reasoning on demand, and a program-aided baseline.

Same generated units and ground truth as suffbench.py (same seed, so the pairs are identical).  Systems:
  copy        keep the stored sentence
  teacher     the 8B compressor as zero-shot two-input operator
  local12     tnet12; at [LEDGER] its own draft              (CoCoMem, local mode)
  L12         tnet12; [LEDGER] sentences verified by the 8B compressor against the facts      (CoCoMem-L)
  T12         the same with the compressor in reasoning mode  (CoCoMem-T)
  ledger_all  8B repair of every sentence from the facts
  think_all   8B reasoning-mode repair of every sentence from the facts (upper reference, costly)
  tool        program-aided baseline: the 8B writes a Python function over the numbers of the corrected facts, a
              program executes it, and the 8B states the computed values in the sentence.  It is given structured
              numeric facts, which the real memory does not have; the benchmark favours it."""
import argparse
import asyncio
import json
import random
import re

from commutator import T_PROMPT3, LEDGER
from cocomem import ledger_call
from llm import LLM, Usage, gather_limited
from suffbench import build, ok, truth, fact, ROOT, NUM

TOOL1 = """Corrected facts, one per line (the numbers are the logged values, in this order):
{facts}
The user corrected: {old} => {new}
Sentence stored before the correction: {sent}

Write a Python function f(vals) where vals is the list of the numbers of the corrected facts in the order above. It returns the list of the numbers the sentence should state after the correction (an empty list if the sentence does not depend on the corrected fact). Use only min, max, sum, len, sorted, abs, round and list operations. Output only the code."""
TOOL2 = """The user corrected: {old} => {new}
Sentence stored before the correction: {sent}
A program computed the values this sentence should state after the correction: {vals}
Rewrite the sentence so that it states these values, keeping its citations. Output only the sentence."""
SAFE = {"min": min, "max": max, "sum": sum, "len": len, "sorted": sorted, "abs": abs, "round": round, "range": range,
        "enumerate": enumerate, "list": list, "reversed": reversed, "zip": zip}


async def op_tool(llm, c, usage):
    facts = [(c["new"] if j == c["i"] else f) for j, f in enumerate(c["facts"])]
    code = await llm.chat(TOOL1.format(facts="\n".join(facts), old=c["old"], new=c["new"], sent=c["sent"]), max_tokens=400, usage=usage)
    code = re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.M).strip()
    vals = [int(NUM.findall(f)[0]) for f in facts]
    try:
        ns = {"__builtins__": SAFE}
        exec(code, ns)
        res = ns["f"](vals)
        res = list(res) if isinstance(res, (list, tuple)) else [res]
    except Exception:
        return c["sent"]
    if not res:
        return c["sent"]
    r = await llm.chat(TOOL2.format(old=c["old"], new=c["new"], sent=c["sent"], vals=res), max_tokens=200, usage=usage)
    return r.strip().split("\n")[0].strip().strip('"') or c["sent"]


async def op_single(T, c, usage):
    r = await T.chat(T_PROMPT3.format(old=f"[c{c['i'] + 1}] {c['old']}", new=c["new"], sent=c["sent"]), max_tokens=200, usage=usage)
    r = r.strip().split("\n")[0].strip().strip('"')
    if r.startswith(LEDGER):
        return True, (r[len(LEDGER):].strip().strip('"') or c["sent"])
    return False, r or c["sent"]


async def op_ledger(llm, c, draft, usage, think=False):
    facts = "\n".join(f"[c{j + 1}] {(c['new'] if j == c['i'] else f)}" for j, f in enumerate(c["facts"]))
    got = await ledger_call(llm, "digest", f"[c{c['i'] + 1}] {c['old']}", f"[c{c['i'] + 1}] {c['new']}", facts, [draft], [0], usage, think=think)
    return got.get(0) or draft


async def op_teacher(llm, c, usage):
    from commutator import T_PROMPT
    r = await llm.chat(T_PROMPT.format(old=f"[c{c['i'] + 1}] {c['old']}", new=c["new"], sent=c["sent"]), max_tokens=160, usage=usage)
    return r.strip().split("\n")[0].strip().strip('"') or c["sent"]


async def run_systems(c, T, llm, U):
    bot, draft = await op_single(T, c, U["tnet12"])
    L12, T12, led, thk, tool, teacher = await asyncio.gather(
        op_ledger(llm, c, draft, U["L12"]) if bot else asyncio.sleep(0, draft),
        op_ledger(llm, c, draft, U["T12"], think=True) if bot else asyncio.sleep(0, draft),
        op_ledger(llm, c, c["sent"], U["ledger_all"]),
        op_ledger(llm, c, c["sent"], U["think_all"], think=True),
        op_tool(llm, c, U["tool"]), op_teacher(llm, c, U["teacher"]))
    return {"bot": bot, "out": {"copy": c["sent"], "teacher": teacher, "local12": draft, "L12": L12, "T12": T12,
                                "ledger_all": led, "think_all": thk, "tool": tool}}


async def chain(units, T, llm, rnd, n):
    names = ("local12", "T12", "think_all", "tool")
    res = {s: [] for s in names}
    jobs = []
    for u in units[:n]:
        i = rnd.randrange(len(u["vals"]))
        seq, cur = [], u["vals"][i]
        for _ in range(3):
            nxt = rnd.choice([x for x in range(3, 80) if x not in u["vals"] and x != cur])
            seq.append(nxt); cur = nxt
        for kind in ("sum", "max", "count"):
            jobs.append((u, i, seq, kind))

    async def one(job):
        u, i, seq, kind = job
        vals = list(u["vals"]); outs = {}
        for sysn in names:
            s, w = u["S"][kind], list(vals)
            for nv in seq:
                c = {"sent": s, "i": i, "old": fact(u["verb"], u["unit"], u["days"][i], w[i]),
                     "new": fact(u["verb"], u["unit"], u["days"][i], nv), "facts": [fact(u["verb"], u["unit"], d, v) for d, v in zip(u["days"], w)]}
                if sysn == "tool":
                    s = await op_tool(llm, c, Usage())
                elif sysn == "think_all":
                    s = await op_ledger(llm, c, s, Usage(), think=True)
                else:
                    bot, dr = await op_single(T, c, Usage())
                    s = await op_ledger(llm, c, dr, Usage(), think=True) if (bot and sysn == "T12") else dr
                w[i] = nv
            lab, exp, _ = truth(kind, vals, i, w[i], u["T"], u["days"])
            stale = sorted({v for v in [truth(kind, vals, i, x, u["T"], u["days"])[1][0] for x in [vals[i]] + seq[:-1]]
                            if isinstance(v, int)} - {exp[0]})
            outs[sysn] = ok({"label": "local" if lab != "copy" else "copy", "sent": u["S"][kind], "exp": exp, "stale": stale}, s)
        return outs

    for o in await gather_limited([one(j) for j in jobs], desc="chain", every=100):
        for k, v in o.items():
            res[k].append(v)
    return {k: 100 * sum(v) / len(v) for k, v in res.items()} | {"n": len(jobs)}


async def main(a):
    cases, units = build(a.units, a.seed)
    T = LLM(a.local, 18432, concurrency=256)
    llm = LLM("qwen3-8b", 18431, concurrency=128)
    U = {k: Usage() for k in ("tnet12", "L12", "T12", "ledger_all", "think_all", "tool", "teacher")}
    R = await gather_limited([run_systems(c, T, llm, U) for c in cases], desc="suff2", every=500)
    out = {"n": len(cases), "labels": {l: sum(c["label"] == l for c in cases) for l in ("copy", "local", "insufficient")}}
    syss = list(R[0]["out"])
    acc = {s: {} for s in syss}
    for s in syss:
        for grp in ("all", "copy", "local", "insufficient"):
            idx = [j for j, c in enumerate(cases) if grp == "all" or c["label"] == grp]
            acc[s][grp] = 100 * sum(ok(cases[j], R[j]["out"][s]) for j in idx) / max(1, len(idx))
        for kind in ("sum", "count", "max", "min", "spread", "second", "other"):
            idx = [j for j, c in enumerate(cases) if c["kind"] == kind]
            acc[s]["k_" + kind] = 100 * sum(ok(cases[j], R[j]["out"][s]) for j in idx) / max(1, len(idx))
    out["acc"] = acc
    y = [c["label"] == "insufficient" for c in cases]; p = [r["bot"] for r in R]
    tp = sum(a_ and b for a_, b in zip(p, y)); P = tp / max(1, sum(p)); Rc = tp / max(1, sum(y))
    y2 = [c["label"] != "copy" for c in cases]
    tp2 = sum(a_ and b for a_, b in zip(p, y2))
    out["detect"] = {"p": P, "r": Rc, "rate": sum(p) / len(p),
                     "rate_on_copy": sum(b for b, c in zip(p, cases) if c["label"] == "copy") / max(1, out["labels"]["copy"]),
                     "rate_on_local": sum(b for b, c in zip(p, cases) if c["label"] == "local") / max(1, out["labels"]["local"]),
                     "p_changing": tp2 / max(1, sum(p)), "r_changing": tp2 / max(1, sum(y2))}
    out["tokens"] = {k: (v.as_dict()["in"] + v.as_dict()["out"]) / len(cases) for k, v in U.items()}
    out["chain"] = await chain(units, T, llm, random.Random(a.seed + 1), a.chain)
    json.dump([{**c, **R[j]} for j, c in enumerate(cases)], open(f"{ROOT}/results/suffbench2_cases.json", "w"))
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=150)
    ap.add_argument("--chain", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local", default="tnet12")
    ap.add_argument("--out", default="suffbench_v12.json")
    asyncio.run(main(ap.parse_args()))
