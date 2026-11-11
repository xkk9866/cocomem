"""How far is one Jacobi step from the corrected fixed point?  (Prop. 2 is exact iff the answer is 0.)

Greedy compression of the corrected inputs I' is the fixed point of u_k = f_k(I'; u_<k).  Starting
from the stored unit u (the draft), step 1 gives J1_k = f_k(I'; u_<k) -- exactly the target the
commutator is trained on (cfgen.py).  Step 2 re-derives every sentence on the step-1 prefix,
J2_k = f_k(I'; J1_<k).  A sentence with J2_k != J1_k is a *prefix-induced* change: it depends on
the correction only through an earlier sentence (the chain u2 = u1 of the reviewer's example).
We report, on the held-out counterfactual set, the share of corrections and of sentences with
such second-order changes, i.e. the empirical violation of prefix insensitivity."""
import asyncio
import json
import re

from cfgen import SAME, chat_prefix, first_sentence, norm, unit_prompt
from cocomem import CoCoMem, split_sentences, strip_cites
from llm import LLM, gather_limited

ROOT = "/data/yaokang/corrmem"
RELEVANT = """A fact in a memory was corrected.
Before: {old}
After: {new}

Two versions of one memory sentence:
A: {a}
B: {b}

Do A and B differ in what they say about this fact, or about anything that follows from it? Differences about other topics do not count. Answer yes or no only."""


async def main():
    llm = LLM("qwen3-8b", 18431, concurrency=160)
    rows = [json.loads(l) for l in open(f"{ROOT}/data/cf_test.jsonl")]
    rows = [r for r in rows if any(r["y"])]
    mems = {}

    async def one(r):
        if r["qid"] not in mems:
            mems[r["qid"]] = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{r['qid']}.json")))
        m = mems[r["qid"]]
        u = m.units.get(r["uid"])
        if u is None:
            return None
        u0 = r["text"]
        rng = split_sentences(u0)
        s0 = [u0[a:b] for a, b in rng]
        if len(s0) != len(r["y"]):
            return None
        j1 = [r["cf_sents"][k] if r["y"][k] else s0[k] for k in range(len(s0))]
        first = min(k for k, y in enumerate(r["y"]) if y)
        pre = chat_prefix(unit_prompt(m, u, {r["cid"]: r["cf_claim"]}))
        sep = [u0[rng[i][1]:rng[i + 1][0]] for i in range(len(rng) - 1)] + [""]
        lead = u0[:rng[0][0]]

        async def step2(k):
            prefix = lead + "".join(j1[i] + sep[i] for i in range(k))
            x = first_sentence(await llm.complete(pre + prefix, max_tokens=len(s0[k]) // 3 + 16))
            if norm(x) == norm(j1[k]):
                return 0
            a = await llm.chat(SAME.format(a=strip_cites(j1[k]), b=strip_cites(x)), max_tokens=4)
            if "no" not in a.lower():
                return 0
            b = await llm.chat(RELEVANT.format(old=r["claim"], new=r["cf_claim"], a=strip_cites(j1[k]), b=strip_cites(x)),
                               max_tokens=4)
            return 2 if "yes" in b.lower() else 1      # 2: the correction propagated through the prefix; 1: drift

        d = await asyncio.gather(*(step2(k) for k in range(first + 1, len(s0))))
        return {"level": r["level"], "n": len(s0), "after": len(d), "second": sum(x > 0 for x in d),
                "second_rel": sum(x == 2 for x in d), "first": sum(r["y"])}

    R = [x for x in await gather_limited([one(r) for r in rows], desc="jacobi2", every=50) if x]
    out = {}
    for lev in ("all", "note", "digest", "profile"):
        X = [x for x in R if lev == "all" or x["level"] == lev]
        if not X:
            continue
        out[lev] = {"corrections": len(X),
                    "corr_with_second_order": 100 * sum(x["second"] > 0 for x in X) / len(X),
                    "second_share_of_changes": 100 * sum(x["second"] for x in X) / max(1, sum(x["first"] + x["second"] for x in X)),
                    "second_per_sentence": 100 * sum(x["second"] for x in X) / max(1, sum(x["after"] for x in X)),
                    "corr_with_second_order_relevant": 100 * sum(x["second_rel"] > 0 for x in X) / len(X),
                    "relevant_per_sentence": 100 * sum(x["second_rel"] for x in X) / max(1, sum(x["after"] for x in X))}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/jacobi2.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
