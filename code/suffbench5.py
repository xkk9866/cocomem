"""Programmatic self-maintainability benchmark (v21).  Ground truth is computed by a program, never by an LLM.

Units state aggregates over 4-6 dated quantitative claims (sum, count above a threshold, maximum, minimum, spread,
runner-up) and one unrelated sentence (suffbench.build).  Classes of a (correction, sentence) pair: independent (copy),
local (repairable from the sentence and the correction: distributive/invertible aggregates), insufficient (holistic under
the correction: the extreme moved away from the corrected claim, spread, runner-up; Lemma 1).

v21: every sentence carries a materialized derivation record (deriv.DERIV, written by the compressor from the unit's
facts, as on the real memories).  The operator reads only (correction, record, sentence) -- no fact is read from the
ledger at repair time -- and maintains the record together with the sentence.
v22 (CM_COMPRESSOR=plan): the records are written first, with the grammar of the memories (annotate_plan); the
benchmark fixes what each sentence states, and a statement enters the benchmark when the record-first write states
the program's value (record.write_ok is the share).

Part A (single corrections): value accuracy by class and kind, tokens per pair.
Part B (sequences): 100 successive corrections of random claims of the same unit, a quarter of them retractions; every
system repairs every sentence after each correction and carries its own record / sentence forward; accuracy after
1, 10, 50 and 100 corrections (Theorem 3).
Part C (record quality): recall of the record's claims against the program's operands.

  python suffbench5.py --students tnet21:18432 --systems copy,comm21,teacher_d,ledger_all,think_all --tag main"""
import argparse
import asyncio
import json
import os
import random
import re

from commutator import T_PROMPT, W_PROMPT, first_line
from commutator21 import D_PROMPT, parse_two, record_ids, out_budget_d, upsert_record, record_entries
from cocomem import ledger_call
from deriv import DERIV, _parse as parse_records
from llm import LLM, Usage, gather_limited
from suffbench import build, fact, ok

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
KINDS = ["sum", "count", "max", "min", "spread", "second", "other"]
STUDENT = {"comm21": "tnet21", "comm21_nd": "tnet21nd", "comm21_w": "tnet21w", "comm21_nv": "tnet21nv", "comm21_nr": "tnet21nr",
           "comm22": "tnet22", "comm22_nd": "tnet22nd", "comm22_w": "tnet22w", "comm22_nv": "tnet22nv", "comm22_nr": "tnet22nr",
           "comm22_raw": "tnet22"}
REC_SYS = {"comm21", "comm21_nv", "comm21_nr", "comm22", "comm22_nv", "comm22_nr", "comm22_raw", "teacher_d"}   # carry (sentence, record)
# Keyed record maintenance (commutator21.upsert_record): the entry of the corrected claim and the computation come from
# the operator, the entries of the other operands are carried over.  comm22_raw stores the operator's whole record.
KEYED = {"comm22", "comm22_nv", "comm22_nr", "teacher_d"}
WINDOWS = [(1, 10), (11, 50), (51, 100)]
_NUM = re.compile(r"\d+")


def operands_ok(rec, vals):
    """Share of the unit's claims whose entry in the record carries the claim's current value."""
    E = {}
    for k, p in record_entries(rec):
        if k:
            E.setdefault(k, []).append(p.split("=", 1)[-1] if "=" in p else p.split(":", 1)[-1])
    return sum(any(str(v) in _NUM.findall(x) for x in E.get(f"c{j + 1}", [])) for j, v in enumerate(vals)) / len(vals)


def entry_ok(rec, i, new):
    """The entry of the corrected claim c_{i+1} in the record carries the value of the corrected fact `new`."""
    v = _NUM.findall(new.split("the user", 1)[-1])
    E = [p.split("=", 1)[-1] if "=" in p else p.split(":", 1)[-1] for k, p in record_entries(rec) if k == f"c{i + 1}"]
    return bool(v) and any(v[0] in _NUM.findall(x) for x in E)


def expected(kind, vals, days, T):
    a = max(range(len(vals)), key=lambda j: vals[j])
    b = min(range(len(vals)), key=lambda j: vals[j])
    r = sorted(range(len(vals)), key=lambda j: -vals[j])[1]
    return {"sum": [sum(vals)], "count": [sum(v > T for v in vals)], "max": [vals[a], days[a]], "min": [vals[b], days[b]],
            "spread": [vals[a] - vals[b]], "second": [vals[r], days[r]], "other": []}[kind]


async def annotate(llm, unit, usage):
    """Write-time derivation records of the six aggregate sentences and the unrelated one (same pass as on real memories)."""
    sents = [unit["S"][k] for k in KINDS]
    facts = "\n".join(f"[c{j + 1}] ({unit['days'][j][:3]}, user) {f}" for j, f in enumerate(unit["facts"]))
    shown = "\n".join(f"({k}) {s}" for k, s in enumerate(sents))
    r = await llm.chat(DERIV.format(kind="digest", facts=facts, unit=shown), max_tokens=900, usage=usage)
    D = parse_records(r, len(sents))
    return dict(zip(KINDS, D))


# Record-first write of a benchmark unit (v22).  The grammar is the one the memories are written with
# (prompts.PLAN_FORMAT): for every sentence the compressor writes what it will state, the record, and the sentence.
# The benchmark fixes what the seven sentences state (their T lines), so that the program knows the value of each;
# the record is the compressor's.  A record enters the benchmark when the sentence the compressor rendered from it
# states the program's value: the benchmark measures the maintenance of correct units, and a write that states a wrong
# value is a write error, reported separately (write_ok).
PLAN_MODE = os.environ.get("CM_COMPRESSOR") == "plan"
INTENT = {"sum": "total {unit} over these days", "count": "number of days with more than {T} {unit}",
          "max": "highest single day and its value", "min": "lowest single day and its value",
          "spread": "difference between the highest and the lowest day", "second": "second-highest day and its value",
          "other": "the user keeps a weekly log of this habit"}
PLAN_BENCH = """Below are facts about one topic, gathered from past conversation sessions with the user. Write a memory digest of them.

Facts:
{facts}

{format}

The digest has exactly {n} sentences. They state the following, in this order; use each line as the T line of its triple:
{intents}

Digest:"""
_TDS = re.compile(r"^\s*\**\s*([TDS])\s*\**\s*:\s*(.*)$", re.I)


async def annotate_plan(llm, unit, usage):
    import prompts
    from cocomem import restrict_record
    facts = "\n".join(f"[c{j + 1}] ({unit['days'][j][:3]}, user) {f}" for j, f in enumerate(unit["facts"]))
    intents = [INTENT[k].format(unit=unit["unit"], T=unit["T"]) for k in KINDS]
    r = await llm.chat(PLAN_BENCH.format(facts=facts, format=prompts.PLAN_FORMAT.replace("{kind}", "digest"), n=len(KINDS),
                                         intents="\n".join("T: " + x for x in intents)), max_tokens=1200, usage=usage)
    tri, cur = [], None
    for line in r.splitlines():
        m = _TDS.match(line)
        if not m:
            continue
        tag, body = m.group(1).upper(), m.group(2).strip()
        if tag == "T":
            cur = {"T": body, "D": "none", "S": ""}
            tri.append(cur)
        elif cur is not None and tag == "D":
            cur["D"] = body or "none"
        elif cur is not None and tag == "S" and not cur["S"]:
            cur["S"] = body
    allowed = {f"c{j + 1}" for j in range(len(unit["facts"]))}
    D, S, good = {}, {}, {}
    for pos, k in enumerate(KINDS):
        t = tri[pos] if len(tri) == len(KINDS) else next((x for x in tri if x["T"].lower().startswith(intents[pos].lower()[:24])), None)
        D[k] = restrict_record(t["D"], allowed) if t else "none"
        S[k] = t["S"] if t else ""
        exp = expected(k, unit["vals"], unit["days"], unit["T"])
        nums = set(re.findall(r"\d+", re.sub(r"\[[^\]]*\]", "", S[k])))
        good[k] = bool(t) and bool(S[k]) and all((str(e) in nums) if isinstance(e, int) else (e in S[k]) for e in exp)
    return {"D": D, "S": S, "ok": good}


def deref(facts, ids, i):
    lines = [f"[c{j + 1}] {facts[j]}" for j in range(len(facts)) if f"c{j + 1}" in ids and j != i]
    return "\n".join(lines) if lines else "(none)"


class Ops:
    def __init__(self, a):
        self.T = {}
        for x in filter(None, a.students.split(",")):
            n, p = x.split(":")
            self.T[n] = LLM(n, int(p), concurrency=256)
        self.llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=128)
        self.U = {}
        self.sys = [s for s in a.systems.split(",") if s not in STUDENT or STUDENT[s] in self.T]

    def u(self, k):
        return self.U.setdefault(k, Usage())

    async def run(self, sysn, sent, rec, i, old, new, facts):
        """One repair step.  facts: current facts BEFORE the correction (facts[i] == old).  Returns (sentence, record)."""
        o = f"[c{i + 1}] {old}"
        if sysn == "copy":
            return sent, rec
        if sysn in ("comm21_nd", "comm22_nd"):
            x = first_line(await self.T[STUDENT[sysn]].chat(T_PROMPT.format(old=o, new=new, sent=sent), max_tokens=160, usage=self.u(sysn)))
            return (x or sent), rec
        if sysn in ("comm21_w", "comm22_w"):
            p = W_PROMPT.format(old=o, new=new, facts=deref(facts, record_ids(rec) | {"c%d" % (j + 1) for j in range(len(facts))
                                                                                     if f"c{j + 1}" in sent}, i), sent=sent)
            x = first_line(await self.T[STUDENT[sysn]].chat(p, max_tokens=160, usage=self.u(sysn)))
            return (x or sent), rec
        if sysn in REC_SYS:
            T = self.llm if sysn == "teacher_d" else self.T[STUDENT[sysn]]
            r = await T.chat(D_PROMPT.format(old=o, new=new, rec=rec, sent=sent), max_tokens=out_budget_d(sent, rec), usage=self.u(sysn))
            rec2, x = parse_two(r, rec, sent)
            rec2 = rec2 or rec
            return (x or sent), (upsert_record(rec, rec2, f"c{i + 1}") if sysn in KEYED else rec2)
        cur = "\n".join(f"[c{j + 1}] {(new if j == i else f)}" for j, f in enumerate(facts))
        got = await ledger_call(self.llm, "digest", o, f"[c{i + 1}] {new}", cur, [sent], [0], self.u(sysn), think=sysn == "think_all")
        return (got.get(0) or sent), rec


async def part_a(ops, cases, units, Ds):
    S = ops.sys

    async def one(c):
        rec = Ds[c["u"]][c["kind"]]
        outs = await asyncio.gather(*(ops.run(s, c["sent"], rec, c["i"], c["facts"][c["i"]], c["new"], c["facts"]) for s in S))
        return {s: o[0] for s, o in zip(S, outs)}, {s: o[1] for s, o in zip(S, outs)}

    R = await gather_limited([one(c) for c in cases], desc="A", every=500)
    acc = {}
    for s in S:
        acc[s] = {}
        for grp in ("all", "copy", "local", "insufficient"):
            idx = [j for j, c in enumerate(cases) if grp == "all" or c["label"] == grp]
            acc[s][grp] = 100 * sum(ok(cases[j], R[j][0][s]) for j in idx) / max(1, len(idx))
        for kind in KINDS:
            idx = [j for j, c in enumerate(cases) if c["kind"] == kind]
            acc[s]["k_" + kind] = 100 * sum(ok(cases[j], R[j][0][s]) for j in idx) / max(1, len(idx))
        if s in REC_SYS:
            # the stored record after the repair: its entry of the corrected claim carries the new value, whether or
            # not the sentence had to change ("copy" = the sentence stays)
            for grp in ("all", "copy", "local", "insufficient"):
                idx = [j for j, c in enumerate(cases) if grp == "all" or c["label"] == grp]
                acc[s]["entry_" + grp] = 100 * sum(entry_ok(R[j][1][s], cases[j]["i"], cases[j]["new"]) for j in idx) / max(1, len(idx))
    tok = {s: (ops.u(s).as_dict()["in"] + ops.u(s).as_dict()["out"]) / len(cases) for s in S if s != "copy"}
    return acc, tok, R


async def part_b(ops, units, Ds, n_units, length, seed, wok=None):
    """Long correction sequences with retractions; every system carries its own sentence (and record) forward."""
    S = [s for s in ops.sys if s != "think_all"]
    CK = [1, 10, 50, 100]
    TRACE, trace = os.environ.get("CM_SUFF_TRACE"), []   # optional: every step of every sequence, for case analysis
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
            if wok is None or wok[ui][kind]:
                jobs.append((ui, kind, seq))

    async def one(job):
        ui, kind, seq = job
        u = units[ui]
        res = {}
        all_ids = {f"c{j + 1}" for j in range(len(u["vals"]))}
        for s in S:
            vals, sent, rec, prev = list(u["vals"]), u["S"][kind], Ds[ui][kind], expected(kind, u["vals"], u["days"], u["T"])
            pts, rq, oq, steps = {}, {}, {}, []
            was_good = True             # a statement enters the benchmark only if its sentence states the program's value
            for t, (i, nv) in enumerate(seq, 1):
                facts = [fact(u["verb"], u["unit"], d, v) for d, v in zip(u["days"], vals)]
                newf = fact(u["verb"], u["unit"], u["days"][i], nv)
                # the premise of Theorem 3: before the step, every operand entry of the carried record is current
                cur = s in REC_SYS and operands_ok(rec, vals) == 1.0
                sent, rec = await ops.run(s, sent, rec, i, facts[i], newf, facts)
                vals[i] = nv
                exp = expected(kind, vals, u["days"], u["T"])
                stale = [x for x in prev if isinstance(x, int) and x not in exp and x not in (u["T"], len(vals))]
                good = bool(ok({"label": "local", "sent": u["S"][kind], "exp": exp, "stale": stale}, sent))
                # (the t-th correction changes what the sentence states, the sentence is right after the t-th repair,
                #  the record was current before the step, the step wrote the entry of the corrected claim right,
                #  number of operands, the sentence stated the current value before the step)
                steps.append((exp != prev, good, cur, s in REC_SYS and entry_ok(rec, i, newf), len(vals), was_good))
                was_good = good
                if TRACE:
                    trace.append({"u": ui, "kind": kind, "s": s, "t": t, "i": i, "new": nv, "exp": exp, "changed": exp != prev,
                                  "good": good, "cur": cur, "sent": sent, "rec": rec if s in REC_SYS else ""})
                if t in CK:
                    pts[t] = good
                    # the carried record after t steps (eta of Theorem 3): rq, it still names every operand and
                    # carries the current value of the corrected claim; oq, the share of the operands whose entry
                    # carries the operand's current value
                    rq[t] = int(record_ids(rec) >= all_ids and str(nv) in re.sub(r"\[[^\]]*\]", "", rec or ""))
                    oq[t] = operands_ok(rec, vals)
                prev = exp
            res[s] = (pts, rq, oq, steps)
        return res

    R = await gather_limited([one(j) for j in jobs], desc="B", every=20)
    if TRACE:
        json.dump(trace, open(TRACE, "w"))
    CK = [t for t in CK if t <= length]
    pct = lambda xs: 100 * sum(xs) / max(1, len(xs))
    out = {s: {str(t): pct([r[s][0][t] for r in R]) for t in CK} for s in S} | {"n": len(jobs), "length": length}
    out["record_ok"] = {s: {str(t): pct([r[s][1][t] for r in R]) for t in CK} for s in S if s in REC_SYS}
    out["operands_ok"] = {s: {str(t): pct([r[s][2][t] for r in R]) for t in CK} for s in S if s in REC_SYS}
    # No accumulation (Theorem 3) is a statement about the repair of the t-th correction, so it is measured on the
    # sentences whose value the t-th correction changes, by position in the sequence.  The state accuracy above also
    # counts the sentences a correction leaves alone, whose share falls along a sequence (the copy row).
    out["step"], out["step_n"], out["step_all"] = {}, {}, {}
    for s in S:
        out["step"][s], out["step_all"][s] = {}, {}
        for a, b in WINDOWS:
            ch = [x[1] for r in R for x in r[s][3][a - 1:b] if x[0]]
            out["step"][s][f"{a}-{b}"] = pct(ch)
            out["step_all"][s][f"{a}-{b}"] = pct([x[1] for r in R for x in r[s][3][a - 1:b]])
            out["step_n"][f"{a}-{b}"] = len(ch)
    # Are errors transient?  The repair of the t-th correction on the sentences it changes, split by the state the
    # sentence was in before the step: it stated the value that was current then (from_right), or it did not, because
    # an earlier repair had failed (from_wrong).  An operator that restates the sentence from its operands repairs
    # both alike; one that leans on the stored result carries an earlier error forward.
    out["recover"] = {}
    for s in S:
        out["recover"][s] = {}
        for a, b in WINDOWS + [(1, length)]:
            ch = [x for r in R for x in r[s][3][a - 1:b] if x[0]]
            right, wrong = [x[1] for x in ch if x[5]], [x[1] for x in ch if not x[5]]
            # corrections that leave the value of the sentence as it was: a right sentence must stay (idle_keep), and
            # an operator that reads the operands again can restate a wrong one (idle_fix)
            idle = [x for r in R for x in r[s][3][a - 1:b] if not x[0]]
            ikeep, ifix = [x[1] for x in idle if x[5]], [x[1] for x in idle if not x[5]]
            out["recover"][s]["all" if (a, b) == (1, length) else f"{a}-{b}"] = {
                "from_right": pct(right), "from_wrong": pct(wrong), "share_right": 100 * len(right) / max(1, len(ch)),
                "n_right": len(right), "n_wrong": len(wrong),
                "idle_keep": pct(ikeep), "idle_fix": pct(ifix), "n_idle_right": len(ikeep), "n_idle_wrong": len(ifix)}
    # The repair of the t-th correction by the aggregate the sentence states (changed sentences), by position
    out["step_kind"] = {}
    for s in S:
        out["step_kind"][s] = {}
        for kind in KINDS[:-1]:
            rows = [r for r, j in zip(R, jobs) if j[1] == kind]
            out["step_kind"][s][kind] = {("all" if (a, b) == (1, length) else f"{a}-{b}"):
                                         pct([x[1] for r in rows for x in r[s][3][a - 1:b] if x[0]])
                                         for a, b in WINDOWS + [(1, length)]} | {"chains": len(rows)}
    # The two quantities of Theorem 3, measured: eps, the error of a repair whose record held every operand's current
    # value before the step (changed sentences); eta, the share of steps that leave the entry of the corrected claim
    # without its new value.  bound = eps + (m - 1) eta with m the mean number of operands; observed = the error on
    # the changed sentences over the whole sequence.
    out["theorem"] = {}
    for s in S:
        if s not in REC_SYS:
            continue
        T = out["theorem"][s] = {}
        for a, b in WINDOWS + [(1, length)]:
            st = [x for r in R for x in r[s][3][a - 1:b]]
            ch = [x for x in st if x[0]]
            cc = [x for x in ch if x[2]]
            eps, eta = 100 - pct([x[1] for x in cc]), 100 - pct([x[3] for x in st])
            m = sum(x[4] for x in ch) / max(1, len(ch))
            # eps on the repairs that start from a current record, by the state of the sentence before the step
            c_right, c_wrong = [x[1] for x in cc if x[5]], [x[1] for x in cc if not x[5]]
            T["all" if (a, b) == (1, length) else f"{a}-{b}"] = {
                "eps": eps, "eta": eta, "current": pct([x[2] for x in ch]), "n_current": len(cc), "m": m,
                "bound": min(100.0, eps + (m - 1) * eta), "observed": 100 - pct([x[1] for x in ch]),
                "eps_right": 100 - pct(c_right), "eps_wrong": 100 - pct(c_wrong),
                "share_right": 100 * len(c_right) / max(1, len(cc))}
    return out


async def main(a):
    cases, units = build(a.units, a.seed)
    ops = Ops(a)
    wu = Usage()
    n_all = len(cases)
    if PLAN_MODE:
        W = await asyncio.gather(*(annotate_plan(ops.llm, u, wu) for u in units))
        Ds, wok = [w["D"] for w in W], [w["ok"] for w in W]
        cases = [c for c in cases if wok[c["u"]][c["kind"]]]
    else:
        Ds = await asyncio.gather(*(annotate(ops.llm, u, wu) for u in units))
        wok = [{k: True for k in KINDS} for _ in units]
    # Part C: record recall against the program's operands (every claim enters every aggregate)
    rec = [len(record_ids(Ds[j][k]) & {f"c{i + 1}" for i in range(len(u["vals"]))}) / len(u["vals"])
           for j, u in enumerate(units) for k in KINDS[:-1]]
    out = {"n": len(cases), "n_all": n_all,
           "labels": {l: sum(c["label"] == l for c in cases) for l in ("copy", "local", "insufficient")},
           "record": {"recall": 100 * sum(rec) / len(rec), "complete": 100 * sum(r == 1 for r in rec) / len(rec),
                      "tokens_per_unit": (wu.as_dict()["in"] + wu.as_dict()["out"]) / len(units),
                      "record_tokens_mean": sum(len(Ds[j][k]) // 4 + 1 for j in range(len(units)) for k in KINDS[:-1]) / (6 * len(units)),
                      "record_first": PLAN_MODE,
                      "write_ok": 100 * sum(wok[j][k] for j in range(len(units)) for k in KINDS[:-1]) / (6 * len(units)),
                      "write_ok_kind": {k: 100 * sum(wok[j][k] for j in range(len(units))) / len(units) for k in KINDS}}}
    out["acc"], out["tokens"], R = await part_a(ops, cases, units, Ds)
    print(json.dumps(out, indent=1), flush=True)
    out["chain"] = await part_b(ops, units, Ds, a.chain_units, a.chain_len, a.seed + 1, wok) if a.chain_len else {}
    json.dump([{**c, "rec": Ds[c["u"]][c["kind"]], "out": R[j][0], "out_rec": R[j][1]} for j, c in enumerate(cases)],
              open(f"{ROOT}/results/suffbench_v21_cases_{a.tag}.json", "w"))
    print(json.dumps(out["chain"], indent=1))
    # one file per run (tag); j_tables.sh merges suffbench_v21_<tag>.json into suffbench_v21.json
    json.dump(out, open(f"{ROOT}/results/suffbench_v21_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain_units", type=int, default=30)
    ap.add_argument("--chain_len", type=int, default=100)
    ap.add_argument("--students", default="tnet21:18432", help="served students name:port,...")
    ap.add_argument("--systems", default="copy,comm21,teacher_d,ledger_all,think_all")
    ap.add_argument("--out", default="suffbench_v21.json")
    ap.add_argument("--tag", default="main")
    asyncio.run(main(ap.parse_args()))
