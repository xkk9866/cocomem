"""Direct test of correction commutativity on the stored memories (not via QA).

For every evidence correction (scenario S1) and every unit u in its anchored closure we compare the
candidate repair R(C(H), delta)_u with the reference C(delta(H))_u, the compressor re-run on the
unit's corrected inputs (`recompress`; on these units it is exactly what a full rebuild computes,
Prop. 1).  Information-level commutativity of a unit holds when
    not stale(cand)                      -- the old value (or anything computed from it) is not stated as current
    and  has(ref) => has(cand)           -- whatever the reference states about the corrected fact, the candidate states
We also report the edit ratio (share of stored sentences rewritten: minimality) and collateral
change (share of units OUTSIDE the closure whose text changed: must be 0).  Judge: Llama-3.1-8B
(int4), a model family that neither builds nor repairs the memories."""
import argparse
import asyncio
import glob
import json
import re

from cocomem import split_sentences
from llm import LLM, gather_limited

ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
STALE = """A user corrected something they had told their assistant.
Earlier (now wrong) information: {old}
The user's correction: "{corr}"

Memory text kept by the assistant:
<<<{text}>>>

Does this memory text still present the earlier, wrong information -- or anything computed from it -- as if it were current? A mention that is clearly marked as earlier, previous or corrected does not count. Answer yes or no only."""
HAS = """A user corrected something they had told their assistant.
Earlier (now wrong) information: {old}
The user's correction: "{corr}"

Memory text kept by the assistant:
<<<{text}>>>

Does this memory text state the corrected information, or something that follows from it? Answer yes or no only."""


FACT = """A user corrected something they had told their assistant.
Earlier (now wrong) information: {old}
The user's correction: "{corr}"

Memory text kept by the assistant:
<<<{text}>>>

Is everything this memory text states about the corrected information, and about anything that follows from it, consistent with the correction? Statements clearly marked as earlier or previous are fine; a text that does not mention the information at all is also fine. Answer yes or no only."""


EQUIV = """A user corrected something they had told their assistant.
Earlier (now wrong) information: {old}
The user's correction: "{corr}"

Two versions of the same memory text:
A: <<<{a}>>>
B: <<<{b}>>>

Do A and B state the same about the corrected information and about anything that follows from it (values, totals, counts, dates, orders, conclusions)? Differences in wording, or in content unrelated to the correction, do not matter. Answer yes or no only."""


def sents(t):
    return [re.sub(r"\s+", " ", t[a:b]).strip() for a, b in split_sentences(t)]


def load(tag, strat):
    out = {}
    for f in glob.glob(f"{ROOT}/results/{tag}/{strat}/*.json"):
        d = json.load(open(f))
        out[d.get("qid") or d.get("cid")] = d
    return out


async def main(a):
    llm = LLM(a.model, a.port, concurrency=96)
    corr = {r["qid"]: r for r in json.load(open(f"{ROOT}/data/lme_corr.json"))}
    ref_all = load(a.tag, a.ref)
    global VALID
    import analyze
    analyze.KEY = "label_qwen25-7b"
    VALID = analyze.valid_set(a.tag)[0]
    yes = lambda r: r.strip().lower().startswith("yes")
    summary = {}
    for strat in a.methods.split(","):
        R = load(a.tag, strat)
        jobs = []
        for q, ref in ref_all.items():
            d = R.get(q)
            if not d or "S1" not in d.get("units", {}) or "S1" not in ref.get("repair", {}):
                continue
            rev, clo = ref["repair"]["S1"].get("revised", []), ref["repair"]["S1"].get("closure", [])
            if not rev or not clo:
                continue
            st = json.load(open(f"{ROOT}/states/cocomem/{q}.json"))
            old = " ".join(st["claims"][c]["text"] for c in rev if c in st["claims"])
            cm = corr[q]["affect"]["correction_message"]
            base, cand, refu = d["units"]["S0"], d["units"]["S1"], ref["units"]["S1"]
            for uid in clo:
                if uid in cand and uid in refu and uid in base:
                    jobs.append(("u", q, uid, old, cm, base[uid], cand[uid], refu[uid]))
            outside = [u for u in base if u not in clo and u in cand]
            jobs.append(("c", q, None, None, None, outside, base, cand))

        async def one(j):
            kind, q, uid, old, cm, b, c, r = j
            if kind == "c":
                outside, base, cand = b, c, r
                return {"q": q, "collateral": sum(cand[u] != base[u] for u in outside) / max(1, len(outside))}
            s0, s1 = sents(b), set(sents(c))
            st, hc, hr, fc, eq = await asyncio.gather(
                llm.chat(STALE.format(old=old, corr=cm, text=c), max_tokens=3),
                llm.chat(HAS.format(old=old, corr=cm, text=c), max_tokens=3),
                llm.chat(HAS.format(old=old, corr=cm, text=r), max_tokens=3),
                llm.chat(FACT.format(old=old, corr=cm, text=c), max_tokens=3),
                llm.chat(EQUIV.format(old=old, corr=cm, a=r, b=c), max_tokens=3) if c != r else asyncio.sleep(0, "yes"))
            stale, has_c, has_r = yes(st), yes(hc), yes(hr)
            return {"q": q, "uid": uid, "stale": stale, "missing": has_r and not has_c, "equiv": yes(eq),
                    "commute": (not stale) and (has_c or not has_r), "correct": yes(fc),
                    "edit": sum(x not in s1 for x in s0) / max(1, len(s0))}

        res = await gather_limited([one(j) for j in jobs], desc=strat, every=1000)
        U = [x for x in res if "uid" in x]
        C = [x["collateral"] for x in res if "collateral" in x]
        mean = lambda v: 100 * sum(v) / max(1, len(v))
        qs = sorted({x["q"] for x in U})
        def part(V):
            qv = sorted({x["q"] for x in V})
            return {"units": len(V), "questions": len(qv), "commute": mean([x["commute"] for x in V]),
                    "correct": mean([x["correct"] for x in V]), "stale": mean([x["stale"] for x in V]),
                    "missing": mean([x["missing"] for x in V]), "equiv": mean([x["equiv"] for x in V]),
                    "q_equiv": mean([all(x["equiv"] for x in V if x["q"] == q) for q in qv]),
                    "q_commute": mean([all(x["commute"] for x in V if x["q"] == q) for q in qv]),
                    "q_correct": mean([all(x["correct"] for x in V if x["q"] == q) for q in qv])}
        summary[strat] = {"valid": part([x for x in U if x["q"] in VALID]), "correct": mean([x["correct"] for x in U]),
                          "q_correct": mean([all(x["correct"] for x in U if x["q"] == q) for q in qs]),
                          "units": len(U), "questions": len(qs),
                          "commute": mean([x["commute"] for x in U]), "stale": mean([x["stale"] for x in U]),
                          "missing": mean([x["missing"] for x in U]), "edit": mean([x["edit"] for x in U]),
                          "collateral": mean(C),
                          "q_commute": mean([all(x["commute"] for x in U if x["q"] == q) for q in qs])}
        print(strat, json.dumps(summary[strat]), flush=True)
        json.dump(res, open(f"{ROOT}/results/comm2_{a.tag}_{strat}.json", "w"))
    json.dump(summary, open(f"{ROOT}/results/comm2_{a.tag}{a.suffix}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main3")
    ap.add_argument("--ref", default="recompress")
    ap.add_argument("--methods", default="append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm_overwrite,recompress,rebuild")
    ap.add_argument("--model", default="llama31")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--port", type=int, default=18431)
    asyncio.run(main(ap.parse_args()))
