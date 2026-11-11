"""Verify-then-defer on the programmatic benchmark (same cases as suffbench2).
The commutator's local draft is checked by the SAME commutation verifier that produced its training targets (the
compressor reads the corrected facts and answers whether the draft closes the square, P(yes) >= 1/2); rejected drafts are
re-derived by the compressor's reasoning-mode ledger repair, starting from the draft.  No second network, no threshold
beyond the verifier's own yes/no decision."""
import argparse
import asyncio
import json
import math

from openai import AsyncOpenAI

from commutator import T_PROMPT, VERIFY
from llm import LLM, Usage, gather_limited
from suffbench import build, ok, ROOT
from suffbench2 import op_ledger

cli = AsyncOpenAI(base_url="http://localhost:18431/v1", api_key="x", timeout=600)
SEM = asyncio.Semaphore(64)


async def p_yes(prompt, usage):
    async with SEM:
        r = await cli.chat.completions.create(model="qwen3-8b", messages=[{"role": "user", "content": prompt}], max_tokens=1,
                                              temperature=0.0, logprobs=True, top_logprobs=10,
                                              extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    usage.add(r.usage.prompt_tokens, r.usage.completion_tokens)
    py = pn = 0.0
    for t in r.choices[0].logprobs.content[0].top_logprobs:
        w = t.token.strip().lower()
        if w == "yes":
            py += math.exp(t.logprob)
        elif w == "no":
            pn += math.exp(t.logprob)
    return py / (py + pn) if py + pn > 0 else 0.0


async def one(c, T, llm, U):
    r = await T.chat(T_PROMPT.format(old=f"[c{c['i'] + 1}] {c['old']}", new=c["new"], sent=c["sent"]), max_tokens=200, usage=U["tnet"])
    draft = r.strip().split("\n")[0].strip().strip('"') or c["sent"]
    facts = "\n".join(f"[c{j + 1}] {(c['new'] if j == c['i'] else f)}" for j, f in enumerate(c["facts"]))
    p = await p_yes(VERIFY.format(facts=facts, old=f"[c{c['i'] + 1}] {c['old']}", new=c["new"], sent=c["sent"], cand=draft), U["verify"])
    final = draft if p >= 0.5 else await op_ledger(llm, c, draft, U["think"], think=True)
    return {"draft": draft, "p": p, "final": final, "defer": p < 0.5}


async def main(a):
    cases, _ = build(a.units, a.seed)
    T = LLM(a.local, 18432, concurrency=256)
    llm = LLM("qwen3-8b", 18431, concurrency=128)
    U = {k: Usage() for k in ("tnet", "verify", "think")}
    R = await gather_limited([one(c, T, llm, U) for c in cases], desc="vtd", every=500)
    out = {"n": len(cases)}
    acc = {}
    for name, key in (("local", "draft"), ("verify_defer", "final")):
        acc[name] = {}
        for grp in ("all", "copy", "local", "insufficient"):
            idx = [j for j, c in enumerate(cases) if grp == "all" or c["label"] == grp]
            acc[name][grp] = 100 * sum(ok(cases[j], R[j][key]) for j in idx) / max(1, len(idx))
        for kind in ("sum", "count", "max", "min", "spread", "second", "other"):
            idx = [j for j, c in enumerate(cases) if c["kind"] == kind]
            acc[name]["k_" + kind] = 100 * sum(ok(cases[j], R[j][key]) for j in idx) / max(1, len(idx))
    out["acc"] = acc
    bad = [not ok(c, r["draft"]) for c, r in zip(cases, R)]
    d = [r["defer"] for r in R]
    tp = sum(x and y for x, y in zip(d, bad))
    out["detect"] = {"p": tp / max(1, sum(d)), "r": tp / max(1, sum(bad)), "rate": sum(d) / len(d),
                     "rate_on_copy": sum(x for x, c in zip(d, cases) if c["label"] == "copy") / max(1, sum(c["label"] == "copy" for c in cases)),
                     "rate_on_insufficient": sum(x for x, c in zip(d, cases) if c["label"] == "insufficient") / max(1, sum(c["label"] == "insufficient" for c in cases))}
    out["tokens"] = {k: (v.as_dict()["in"] + v.as_dict()["out"]) / len(cases) for k, v in U.items()}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local", default="tnet10")
    ap.add_argument("--out", default="suffbench_vtd.json")
    asyncio.run(main(ap.parse_args()))
