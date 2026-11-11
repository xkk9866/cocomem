"""Programmatic self-maintainability benchmark (v20).  Ground truth is computed by a program, never by an LLM.

Units state aggregates over 4-6 dated quantitative claims (sum, count above a threshold, maximum, minimum, spread,
runner-up) and one unrelated sentence.  The aggregates span the classical classes of incremental maintenance:
distributive and invertible (sum, count: repairable from the sentence and the correction), and holistic under a
correction (max/min when the corrected claim held the extreme and moved away, spread, runner-up: NOT repairable
from the sentence; Lemma 1).  With the derivation witness every class becomes self-maintainable (Theorem 1).

Part A (single corrections, 2,030 pairs as in v10): value accuracy by ground-truth class and by kind, tokens per pair.
Part B (long sequences): 100 successive corrections of random claims of the same unit, a quarter of them retractions
(the claim returns to an earlier value), every sentence repaired after each correction; accuracy after 1, 10, 50
and 100 corrections (Theorem 3: with a dereferenced witness the error does not accumulate).
Part C (witness quality): recall of the write-time witness against the program's true operands.

  python suffbench4.py --out suffbench_v20.json"""
import argparse
import asyncio
import json
import random
import re

from commutator import T_PROMPT, W_PROMPT, first_line
from cocomem import ledger_call
from llm import LLM, Usage, gather_limited
from suffbench import DAYS, DOM, build, fact, ok, sentences, truth
from witness import WITNESS, _parse

ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
KINDS = ["sum", "count", "max", "min", "spread", "second", "other"]


def expected(kind, vals, days, T):
    """Numbers / day names the sentence must state for the current values."""
    a = max(range(len(vals)), key=lambda j: vals[j])
    b = min(range(len(vals)), key=lambda j: vals[j])
    r = sorted(range(len(vals)), key=lambda j: -vals[j])[1]
    return {"sum": [sum(vals)], "count": [sum(v > T for v in vals)], "max": [vals[a], days[a]], "min": [vals[b], days[b]],
            "spread": [vals[a] - vals[b]], "second": [vals[r], days[r]], "other": []}[kind]


async def annotate(llm, unit, usage):
    sents = [unit["S"][k] for k in KINDS]
    facts = "\n".join(f"[c{j + 1}] {f}" for j, f in enumerate(unit["facts"]))
    shown = "\n".join(f"({k}) {s}" for k, s in enumerate(sents))
    r = await llm.chat(WITNESS.format(kind="digest", facts=facts, unit=shown), max_tokens=600, usage=usage)
    allowed = {f"c{j + 1}" for j in range(len(unit["facts"]))}
    W = _parse(r, len(sents), allowed)
    cites = [re.findall(r"c\d+", " ".join(re.findall(r"\[([^\]]*)\]", s))) for s in sents]
    return {k: list(dict.fromkeys([c for c in ci if c in allowed] + w)) for k, ci, w in zip(KINDS, cites, W)}


def wfacts(facts, ids, i):
    lines = [f"[c{j + 1}] {facts[j]}" for j in range(len(facts)) if f"c{j + 1}" in ids and j != i]
    return "\n".join(lines) if lines else "(none)"


STUDENT = {"comm20": "tnet20", "comm20_nw": "tnet20nw", "comm20_nv": "tnet20nv", "comm20_nr": "tnet20nr"}


class Ops:
    def __init__(self, a):
        self.T = {}
        for x in filter(None, a.students.split(",")):
            n, p = x.split(":")
            self.T[n] = LLM(n, int(p), concurrency=256)
        self.llm = LLM("qwen3-8b", 18431, concurrency=128)
        self.U = {}
        self.sys = [s for s in a.systems.split(",") if s not in STUDENT or STUDENT[s] in self.T]

    def u(self, k):
        return self.U.setdefault(k, Usage())

    async def run(self, sysn, sent, i, old, new, facts, W):
        """facts: current facts BEFORE the correction (facts[i] == old); W: witness ids of the sentence."""
        o = f"[c{i + 1}] {old}"
        if sysn == "copy":
            return sent
        if sysn == "comm20_nw":
            return first_line(await self.T["tnet20nw"].chat(T_PROMPT.format(old=o, new=new, sent=sent), max_tokens=160,
                                                            usage=self.u(sysn))) or sent
        if sysn in STUDENT or sysn == "teacher_w":
            p = W_PROMPT.format(old=o, new=new, facts=wfacts(facts, W, i), sent=sent)
            T = self.llm if sysn == "teacher_w" else self.T[STUDENT[sysn]]
            return first_line(await T.chat(p, max_tokens=160, usage=self.u(sysn))) or sent
        cur = "\n".join(f"[c{j + 1}] {(new if j == i else f)}" for j, f in enumerate(facts))
        got = await ledger_call(self.llm, "digest", o, f"[c{i + 1}] {new}", cur, [sent], [0], self.u(sysn),
                                think=sysn == "think_all")
        return got.get(0) or sent


def systems(ops):
    return ops.sys


async def part_a(ops, cases, units, Ws):
    S = systems(ops)

    async def one(c):
        u = units[c["u"]]
        W = Ws[c["u"]][c["kind"]]
        outs = await asyncio.gather(*(ops.run(s, c["sent"], c["i"], c["facts"][c["i"]], c["new"], c["facts"], W) for s in S))
        return dict(zip(S, outs))

    R = await gather_limited([one(c) for c in cases], desc="A", every=500)
    acc = {}
    for s in S:
        acc[s] = {}
        for grp in ("all", "copy", "local", "insufficient"):
            idx = [j for j, c in enumerate(cases) if grp == "all" or c["label"] == grp]
            acc[s][grp] = 100 * sum(ok(cases[j], R[j][s]) for j in idx) / max(1, len(idx))
        for kind in KINDS:
            idx = [j for j, c in enumerate(cases) if c["kind"] == kind]
            acc[s]["k_" + kind] = 100 * sum(ok(cases[j], R[j][s]) for j in idx) / max(1, len(idx))
    tok = {s: (ops.u(s).as_dict()["in"] + ops.u(s).as_dict()["out"]) / len(cases) for s in S if s != "copy"}
    return acc, tok, R


async def part_b(ops, units, Ws, n_units, length, seed):
    """Long correction sequences with retractions; every sentence is repaired after each correction."""
    S = [s for s in systems(ops) if s in ("comm20", "comm20_nw", "comm20_nv", "comm20_nr", "ledger_all", "copy")]
    CK = [1, 10, 50, 100]
    rnd = random.Random(seed)
    jobs = []
    for ui, u in enumerate(units[:n_units]):
        hist = {i: [v] for i, v in enumerate(u["vals"])}
        seq, vals = [], list(u["vals"])
        for _ in range(length):
            i = rnd.randrange(len(vals))
            back = [v for v in hist[i] if v != vals[i] and v not in vals]
            if back and rnd.random() < 0.25:
                nv = rnd.choice(back)                     # retraction: back to an earlier value of this claim
            else:
                nv = rnd.choice([x for x in range(3, 80) if x not in vals])
            seq.append((i, nv))
            vals[i] = nv
            hist[i].append(nv)
        for kind in KINDS[:-1]:
            jobs.append((ui, kind, seq))

    async def one(job):
        ui, kind, seq = job
        u = units[ui]
        res = {}
        for s in S:
            vals, sent, prev = list(u["vals"]), u["S"][kind], expected(kind, u["vals"], u["days"], u["T"])
            pts = {}
            for t, (i, nv) in enumerate(seq, 1):
                facts = [fact(u["verb"], u["unit"], d, v) for d, v in zip(u["days"], vals)]
                sent = await ops.run(s, sent, i, facts[i], fact(u["verb"], u["unit"], u["days"][i], nv), facts, Ws[ui][kind])
                vals[i] = nv
                exp = expected(kind, vals, u["days"], u["T"])
                if t in CK:
                    stale = [x for x in prev if isinstance(x, int) and x not in exp and x not in (u["T"], len(vals))]
                    pts[t] = ok({"label": "local", "sent": u["S"][kind], "exp": exp, "stale": stale}, sent)
                prev = exp
            res[s] = pts
        return res

    R = await gather_limited([one(j) for j in jobs], desc="B", every=20)
    return {s: {str(t): 100 * sum(r[s][t] for r in R) / len(R) for t in CK} for s in S} | {"n": len(jobs), "length": length}


async def main(a):
    cases, units = build(a.units, a.seed)
    ops = Ops(a)
    wu = Usage()
    Ws = await asyncio.gather(*(annotate(ops.llm, u, wu) for u in units))
    # Part C: witness recall against the program's operands (every claim enters every aggregate)
    rec = [len(set(Ws[j][k]) & {f"c{i + 1}" for i in range(len(u["vals"]))}) / len(u["vals"])
           for j, u in enumerate(units) for k in KINDS[:-1]]
    out = {"n": len(cases), "labels": {l: sum(c["label"] == l for c in cases) for l in ("copy", "local", "insufficient")},
           "witness": {"recall": 100 * sum(rec) / len(rec), "complete": 100 * sum(r == 1 for r in rec) / len(rec),
                       "tokens_per_unit": (wu.as_dict()["in"] + wu.as_dict()["out"]) / len(units)}}
    out["acc"], out["tokens"], R = await part_a(ops, cases, units, Ws)
    print(json.dumps(out, indent=1), flush=True)
    out["chain"] = await part_b(ops, units, Ws, a.chain_units, a.chain_len, a.seed + 1) if a.chain_len else {}
    json.dump([{**c, "out": R[j]} for j, c in enumerate(cases)], open(f"{ROOT}/results/suffbench_v20_cases_{a.tag}.json", "w"))
    print(json.dumps(out["chain"], indent=1))
    import os
    path = f"{ROOT}/results/{a.out}"
    if os.path.exists(path):                    # runs with different served students are merged into one file
        prev = json.load(open(path))
        for key in ("acc", "tokens"):
            out[key] = prev.get(key, {}) | out[key]
        out["chain"] = {k: v for k, v in prev.get("chain", {}).items()} | out["chain"]
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain_units", type=int, default=30)
    ap.add_argument("--chain_len", type=int, default=100)
    ap.add_argument("--students", default="tnet20:18432", help="served students name:port,...")
    ap.add_argument("--systems", default="copy,comm20,comm20_nw,comm20_nv,comm20_nr,teacher_w,ledger_all,think_all")
    ap.add_argument("--out", default="suffbench_v20.json")
    ap.add_argument("--tag", default="a")
    asyncio.run(main(ap.parse_args()))
