"""Direct evaluation of correction commutativity on the stored memories (not via QA).

Reference: anchored recompression of the affected units, which equals C(delta(H)) on those units
(Prop. 1).  For every strategy and every affected unit u of an evidence correction (scenario S1):
  stale     -- does u still present the old information (or anything computed from it) as current?
  agree     -- does u state the same corrected information as the reference unit?
  edit      -- fraction of u's original sentences that were rewritten (minimality)
  collateral-- fraction of NON-affected units whose text changed at all (should be 0)
The judge is a third model family (Llama-3.1-8B-Instruct, int4)."""
import argparse, asyncio, glob, json, os, re
from llm import LLM, gather_limited
from cocomem import split_sentences
from data import load_lme

ROOT = "/data/yaokang/corrmem"
STALE = """A user corrected something they had told their assistant.
Earlier (now wrong) information: {old}
The user's correction: "{corr}"

Memory text kept by the assistant:
\"\"\"{text}\"\"\"

Does this memory text still present the earlier, wrong information -- or anything computed from it -- as if it were current? A mention that is clearly marked as earlier, previous or corrected does not count. Answer yes or no only."""
AGREE = """A user corrected something they had told their assistant: "{corr}"

Reference memory text (written from the corrected information):
\"\"\"{ref}\"\"\"

Candidate memory text:
\"\"\"{text}\"\"\"

Considering only the corrected information and anything that depends on it, does the candidate state the same facts as the reference? Ignore wording and any extra history notes. Answer yes or no only."""


def sents(t):
    return [re.sub(r"\s+", " ", t[a:b]).strip() for a, b in split_sentences(t)]


async def main(a):
    llm = LLM(a.model, 18431, concurrency=160)
    corr = {r["qid"]: r for r in json.load(open(f"{ROOT}/data/lme_corr.json"))}
    states = f"{ROOT}/states/cocomem"
    ref_all = {json.load(open(f))["qid"]: json.load(open(f)) for f in glob.glob(f"{ROOT}/results/{a.tag}/{a.ref}/*.json")}
    out = {}
    for strat in a.methods.split(","):
        R = {json.load(open(f))["qid"]: json.load(open(f)) for f in glob.glob(f"{ROOT}/results/{a.tag}/{strat}/*.json")}
        jobs = []
        for q, ref in ref_all.items():
            d = R.get(q)
            if not d or "units" not in d or "S1" not in d["units"] or "S1" not in ref.get("repair", {}):
                continue
            rev = ref["repair"]["S1"].get("revised", [])
            clo = ref["repair"]["S1"].get("closure", [])
            if not rev or not clo:
                continue
            st = json.load(open(f"{states}/{q}.json"))
            old = " ".join(st["claims"][c]["text"] for c in rev if c in st["claims"])
            cm = corr[q]["affect"]["correction_message"]
            base, cand, refu = d["units"]["S0"], d["units"]["S1"], ref["units"]["S1"]
            for uid in clo:
                if uid in cand and uid in refu and uid in base:
                    jobs.append((q, uid, old, cm, base[uid], cand[uid], refu[uid]))
            others = [u for u in base if u not in clo]
            jobs.append((q, None, None, None, others, base, cand))

        async def one(j):
            q, uid, old, cm, b, c, r = j
            if uid is None:
                ch = [u for u in b if u in c and c[u] != r.get(u)]
                return {"q": q, "collateral": len(ch) / max(1, len(b))}
            s0, s1 = sents(b), set(sents(c))
            edit = sum(x not in s1 for x in s0) / max(1, len(s0))
            r1 = await llm.chat(STALE.format(old=old, corr=cm, text=c), max_tokens=4)
            if c.strip() == r.strip():
                agree = True
            else:
                r2 = await llm.chat(AGREE.format(corr=cm, ref=r, text=c), max_tokens=4)
                agree = "yes" in r2.lower()
            return {"q": q, "uid": uid, "stale": "yes" in r1.lower(), "agree": agree, "edit": edit}

        res = await gather_limited([one(j) for j in jobs], desc=strat, every=500)
        U = [x for x in res if "uid" in x]
        C = [x["collateral"] for x in res if "collateral" in x]
        mean = lambda v: 100 * sum(v) / max(1, len(v))
        out[strat] = {"units": len(U), "stale": mean([x["stale"] for x in U]), "agree": mean([x["agree"] for x in U]),
                      "edit": mean([x["edit"] for x in U]), "collateral": mean(C),
                      "per_q_stale_free": mean([all(not x["stale"] for x in U if x["q"] == q) for q in {x["q"] for x in U}])}
        print(strat, json.dumps(out[strat]), flush=True)
        json.dump(res, open(f"{ROOT}/results/comm_{a.tag}_{strat}.json", "w"))
    json.dump(out, open(f"{ROOT}/results/comm_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main3")
    ap.add_argument("--ref", default="recompress")
    ap.add_argument("--methods", default="append,slots,simupdate,self_span,coarse,ncf,ccf,ccf_overwrite,spec,rebuild,recompress")
    ap.add_argument("--model", default="llama31")
    asyncio.run(main(ap.parse_args()))
