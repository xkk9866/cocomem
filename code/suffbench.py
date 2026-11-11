"""Verifiable local-sufficiency benchmark.

Units are written from a small set of quantitative claims; each sentence states an aggregate whose corrected value
is computed by a program, not by an LLM.  For every (correction, sentence) pair the ground truth says whether the
sentence is independent of the correction (copy), determines its own repair (local: sums, counts over a stated
threshold, a maximum that the corrected value overtakes), or does not (insufficient: the corrected value was the
maximum / minimum and decreased / increased, a spread, a runner-up).  We measure the value accuracy of each
operator and the precision / recall of the commutator's insufficiency verdict, and we apply three successive
corrections of the same claim (version chains) to test that repeated corrections stay exact."""
import argparse
import asyncio
import json
import random
import re

from cocomem import ledger_call
from commutator import T_PROMPT, T_PROMPT3, LEDGER, norm
from llm import LLM, Usage, gather_limited

ROOT = "/data/yaokang/corrmem"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DOM = [("ran", "km"), ("spent", "dollars on groceries"), ("read", "pages"), ("worked", "hours overtime"),
       ("swam", "laps"), ("walked", "thousand steps"), ("practiced piano for", "minutes"), ("drove", "miles")]


def fact(verb, unit, d, v):
    return f"On {d}, the user {verb} {v} {unit}."


def sentences(verb, unit, days, vals, T):
    cites = "[" + ", ".join(f"c{i + 1}" for i in range(len(days))) + "]"
    a = max(range(len(vals)), key=lambda i: vals[i])
    b = min(range(len(vals)), key=lambda i: vals[i])
    r = sorted(range(len(vals)), key=lambda i: -vals[i])[1]
    return {
        "sum": f"Over these days the user {verb} {sum(vals)} {unit} in total {cites}.",
        "count": f"The user {verb} more than {T} {unit} on {sum(v > T for v in vals)} of the {len(vals)} days {cites}.",
        "max": f"The most the user {verb} on a single day was {vals[a]} {unit}, on {days[a]} {cites}.",
        "min": f"The least the user {verb} on a single day was {vals[b]} {unit}, on {days[b]} {cites}.",
        "spread": f"Between the busiest and the quietest day there was a difference of {vals[a] - vals[b]} {unit} {cites}.",
        "second": f"The user's second-highest day was {days[r]}, with {vals[r]} {unit} {cites}.",
        "other": f"The user mentioned that they keep a weekly log of this habit [c1].",
    }


def truth(kind, vals, i, new, T, days):
    """(label, expected numbers / day names, stale numbers) for sentence `kind` after vals[i] := new."""
    w = list(vals); w[i] = new
    a0 = max(range(len(vals)), key=lambda j: vals[j]); b0 = min(range(len(vals)), key=lambda j: vals[j])
    a1 = max(range(len(w)), key=lambda j: w[j]); b1 = min(range(len(w)), key=lambda j: w[j])
    r0 = sorted(range(len(vals)), key=lambda j: -vals[j])[1]; r1 = sorted(range(len(w)), key=lambda j: -w[j])[1]
    if kind == "other":
        return "copy", [], []
    if kind == "sum":
        return "local", [sum(w)], [sum(vals)]
    if kind == "count":
        o, n = sum(v > T for v in vals), sum(v > T for v in w)
        return ("copy" if o == n else "local"), [n], ([] if o == n else [o])
    if kind in ("max", "min"):
        x0, x1, k0, k1 = (vals[a0], w[a1], a0, a1) if kind == "max" else (vals[b0], w[b1], b0, b1)
        if x0 == x1 and k0 == k1:
            return "copy", [x1], []
        own = (i == k0)
        lab = "insufficient" if own and k1 != i else "local"     # the extreme moved away from the corrected day
        return lab, [x1, days[k1]], [x0] if x0 != x1 else []
    if kind == "spread":
        d0, d1 = vals[a0] - vals[b0], w[a1] - w[b1]
        if d0 == d1:
            return "copy", [d1], []
        return "insufficient", [d1], [d0]
    if kind == "second":
        if r0 == r1 and vals[r0] == w[r1]:
            return "copy", [w[r1]], []
        return "insufficient", [w[r1], days[r1]], [vals[r0]] if vals[r0] != w[r1] else []


def build(n_units, seed=0):
    rnd = random.Random(seed)
    cases, units = [], []
    for u in range(n_units):
        verb, unit = rnd.choice(DOM)
        k = rnd.randint(4, 6)
        days = DAYS[:k]
        vals = rnd.sample(range(3, 60), k)
        T = sorted(vals)[k // 2]
        S = sentences(verb, unit, days, vals, T)
        facts = [fact(verb, unit, d, v) for d, v in zip(days, vals)]
        for _ in range(2):
            mode = rnd.choice(["up", "down", "top", "bottom", "dethrone", "lift"])
            i = (vals.index(max(vals)) if mode == "dethrone" else vals.index(min(vals)) if mode == "lift"
                 else rnd.randrange(k))
            if mode == "dethrone":
                mode = "down"
            if mode == "lift":
                mode = "up"
            if mode == "top":
                new = max(vals) + rnd.randint(1, 15)
            elif mode == "bottom":
                new = max(1, min(vals) - rnd.randint(1, 2))
            else:
                new = vals[i] + (rnd.randint(1, 20) if mode == "up" else -rnd.randint(1, vals[i] - 1))
            if new in vals or new < 1:
                continue
            for kind, s in S.items():
                lab, exp, stale = truth(kind, vals, i, new, T, days)
                cases.append({"u": u, "kind": kind, "sent": s, "i": i, "old": facts[i], "new": fact(verb, unit, days[i], new),
                              "facts": facts, "label": lab, "exp": exp, "stale": stale})
        units.append({"verb": verb, "unit": unit, "days": days, "vals": vals, "T": T, "S": S, "facts": facts})
    return cases, units


NUM = re.compile(r"\d+")


def ok(case, out):
    if case["label"] == "copy":
        strip = lambda t: set(NUM.findall(re.sub(r"\[[^\]]*\]", "", t)))
        return norm(out) == norm(case["sent"]) or strip(out) == strip(case["sent"])
    nums = set(NUM.findall(re.sub(r"\[[^\]]*\]", "", out)))
    for e in case["exp"]:
        if isinstance(e, int) and str(e) not in nums:
            return False
        if isinstance(e, str) and e not in out:
            return False
    return not any(str(x) in nums for x in case["stale"])


async def op_local(T, c, usage):
    r = await T.chat(T_PROMPT.format(old=f"[c{c['i'] + 1}] {c['old']}", new=c["new"], sent=c["sent"]), max_tokens=160, usage=usage)
    return r.strip().split("\n")[0].strip().strip('"') or c["sent"]


async def op_suff(T3, c, usage):
    r = await T3.chat(T_PROMPT3.format(old=f"[c{c['i'] + 1}] {c['old']}", new=c["new"], sent=c["sent"]), max_tokens=160, usage=usage)
    return r.strip().split("\n")[0].strip().strip('"')


async def op_ledger(llm, c, draft, usage):
    facts = "\n".join(f"[c{j + 1}] {(c['new'] if j == c['i'] else f)}" for j, f in enumerate(c["facts"]))
    got = await ledger_call(llm, "digest", f"[c{c['i'] + 1}] {c['old']}", f"[c{c['i'] + 1}] {c['new']}", facts, [draft], [0], usage)
    return got.get(0) or draft


async def run_systems(c, T2, T3, llm, U):
    loc, suf = await asyncio.gather(op_local(T2, c, U["tnet2"]), op_suff(T3, c, U["tnet3"]))
    bot = suf.startswith(LEDGER)
    teacher, ledall = await asyncio.gather(op_local(llm, c, U["teacher"]), op_ledger(llm, c, c["sent"], U["ledger_all"]))
    v7 = await op_ledger(llm, c, loc, U["v7"]) if bot else loc
    v6 = await op_ledger(llm, c, c["sent"], U["v6"]) if bot else suf
    return {"bot": bot, "out": {"copy": c["sent"], "teacher": teacher, "local": loc, "suff_nofallback": c["sent"] if bot else suf,
                                "v6": v6, "ledger_all": ledall, "v7": v7}}


async def chain(units, T2, T3, llm, rnd, n):
    """Three successive corrections of the same claim; the sentence is repaired after each one."""
    res = {s: [] for s in ("local", "v7", "ledger_all", "teacher")}
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
        for sysn in res:
            s, w = u["S"][kind], list(vals)
            for nv in seq:
                c = {"sent": s, "i": i, "old": fact(u["verb"], u["unit"], u["days"][i], w[i]),
                     "new": fact(u["verb"], u["unit"], u["days"][i], nv), "facts": [fact(u["verb"], u["unit"], d, v) for d, v in zip(u["days"], w)]}
                if sysn == "local":
                    s = await op_local(T2, c, Usage())
                elif sysn == "teacher":
                    s = await op_local(llm, c, Usage())
                elif sysn == "ledger_all":
                    s = await op_ledger(llm, c, s, Usage())
                else:
                    loc, suf = await asyncio.gather(op_local(T2, c, Usage()), op_suff(T3, c, Usage()))
                    s = await op_ledger(llm, c, loc, Usage()) if suf.startswith(LEDGER) else loc
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
    T2, T3 = LLM(a.local, 18432, concurrency=256), LLM("tnet3", 18433, concurrency=256)
    llm = LLM("qwen3-8b", 18431, concurrency=128)
    U = {k: Usage() for k in ("tnet2", "tnet3", "teacher", "ledger_all", "v6", "v7")}
    R = await gather_limited([run_systems(c, T2, T3, llm, U) for c in cases], desc="suff", every=500)
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
    out["detect"] = {"p": P, "r": Rc, "f1": 2 * P * Rc / max(1e-9, P + Rc), "rate": sum(p) / len(p),
                     "rate_on_copy": sum(b for b, c in zip(p, cases) if c["label"] == "copy") / max(1, out["labels"]["copy"]),
                     "rate_on_local": sum(b for b, c in zip(p, cases) if c["label"] == "local") / max(1, out["labels"]["local"])}
    out["tokens"] = {k: (v.as_dict()["in"] + v.as_dict()["out"]) / len(cases) for k, v in U.items()}
    out["chain"] = await chain(units, T2, T3, llm, random.Random(a.seed + 1), a.chain)
    json.dump([{**c, **R[j]} for j, c in enumerate(cases)], open(f"{ROOT}/results/suffbench_cases.json", "w"))
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=150)
    ap.add_argument("--chain", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local", default="tnet2")
    ap.add_argument("--out", default="suffbench.json")
    asyncio.run(main(ap.parse_args()))
