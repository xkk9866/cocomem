"""Direct test of correction commutativity, version 3: ONE exclusive verdict per stored unit.

comm_eval2 asked three separate yes/no questions (stale / has / consistent) and reported them as separate rates; with an
8B judge they contradict each other (a unit could be 'consistent' and 'stale' at once).  Here the judge answers once:
    A  the unit states the corrected information (or what follows) and presents nothing of the old as current
    B  the unit still presents the old information, or anything computed from it, as current
    C  the unit states nothing about it
so that  Stale = [B]  and  Fresh = [A or C]  add up to 100 by construction.  A unit is *sound* when it is not stale
and loses nothing the reference states (reference = A and candidate = C is a missing statement).
A judge-free surface check is added: a token of the old claim (a number or a name that the correction message does not
repeat) that the unit contained before the correction and still contains after it ('leak').
Run once per judge (--model); agg_eval3.py takes the majority verdict of the three judges."""
import argparse
import asyncio
import glob
import json
import re

from cocomem import split_sentences
from llm import LLM, gather_limited

ROOT = "/data/yaokang/corrmem"
VERDICT = """A user corrected something they had told their assistant.
Earlier (now wrong) information: {old}
The user's correction: "{corr}"

Memory text kept by the assistant:
<<<{text}>>>

Which statement is true of this memory text with respect to the corrected information?
A. It states the corrected information, or something that follows from it, and presents nothing of the earlier information as current.
B. It still presents the earlier, wrong information, or anything computed from it, as if it were current (even if it also states the corrected information). A mention clearly marked as earlier or previous does not count.
C. It states nothing about this information.
Answer with one letter."""

TOK = re.compile(r"\d+(?:[.,]\d+)?|\b[A-Z][a-zA-Z]{2,}\b")


def toks(t):
    return {x.lower() for x in TOK.findall(t)}


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
    letter = lambda r: next((c for c in r.strip().upper() if c in "ABC"), "C")
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
                    jobs.append((q, uid, old, cm, base[uid], cand[uid], refu[uid]))

        async def one(j):
            q, uid, old, cm, b, c, r = j
            vc, vr = await asyncio.gather(
                llm.chat(VERDICT.format(old=old, corr=cm, text=c), max_tokens=3),
                llm.chat(VERDICT.format(old=old, corr=cm, text=r), max_tokens=3))
            vc, vr = letter(vc), letter(vr)
            old_only = toks(old) - toks(cm)
            elig = bool(old_only & toks(b))
            return {"q": q, "uid": uid, "v": vc, "vref": vr, "elig": elig, "leak": elig and bool(old_only & toks(c))}

        res = await gather_limited([one(j) for j in jobs], desc=strat, every=1000)
        json.dump(res, open(f"{ROOT}/results/comm3_{a.tag}_{strat}_{a.model}.json", "w"))
        st_ = 100 * sum(x["v"] == "B" for x in res) / len(res)
        print(strat, len(res), "stale", round(st_, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main3")
    ap.add_argument("--ref", default="recompress")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--model", default="llama31")
    ap.add_argument("--port", type=int, default=18431)
    asyncio.run(main(ap.parse_args()))
