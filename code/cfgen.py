"""Counterfactual interventions on the compressor (self-supervision for the Neural Commutator Field).

For a memory unit u = f(S) and an input claim c in S we apply the correction operator itself:
    delta_c : c -> c'   (c' = c with its key detail changed, produced by the same LLM)
and ask the *same* compressor, teacher-forced on the factual prefix u_{<k}, to write sentence k
under the counterfactual input.  The commutator of compression and correction on u is the set of
sentences whose counterfactual continuation differs from the factual one:
    y_k(u, c) = 1[ f_k(S[c -> c'] ; u_{<k})  !=  f_k(S ; u_{<k}) ]   (greedy decoding)
Teacher forcing on the factual prefix isolates the local effect of c on each sentence (no
divergence cascade); "!=" is semantic (same compressor asked whether the two sentences state the
same information), so wording jitter of greedy decoding near ties is not counted as dependence.
These labels are exactly the support that a correction of c must regenerate for commutativity;
no human annotation and no access to the compressor's internals are needed."""
import argparse
import asyncio
import difflib
import json
import os
import random
import re

import prompts
from cocomem import PROFILE, CoCoMem, split_sentences, strip_cites
from llm import LLM, gather_limited
from splits import add_test200, make_splits

STATES = "/data/yaokang/corrmem/states/cocomem"
PER_LEVEL = {"note": 4, "digest": 6, "profile": 8}
MAXTOK = {"note": 400, "digest": 600, "profile": 900}
NOTE_STRIDE = 3   # every third session note per memory (notes are numerous and homogeneous)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", strip_cites(s).lower()).strip()


def unit_prompt(m, u, override=None):
    claims = m.claims
    saved = {}
    if override:
        for c, t in override.items():
            saved[c] = claims[c]["text"]
            claims[c]["text"] = t
    try:
        facts = m._fact_lines([c for c in u["inputs"] if c in claims])
        if u["level"] == "note":
            return prompts.NOTE.format(date=u["date"], words=m.note_words, facts=facts)
        if u["level"] == "digest":
            return prompts.DIGEST.format(words=m.digest_words, facts=facts)
        if u.get("person"):
            return prompts.PROFILE_PERSON.format(person=u["person"], words=m.profile_words, facts=facts)
        return PROFILE.format(words=m.profile_words, facts=facts)
    finally:
        for c, t in saved.items():
            claims[c]["text"] = t


_TOK = None


def chat_prefix(prompt):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("/home/yaokang/models/Qwen3-8B-AWQ")
    return _TOK.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)


def first_sentence(t):
    t = t.lstrip()
    r = split_sentences(t)
    return t[r[0][0]:r[0][1]].strip() if r else ""


def sent_ratios(u0, u1):
    s0 = [u0[a:b] for a, b in split_sentences(u0)]
    s1 = [norm(u1[a:b]) for a, b in split_sentences(u1)]
    out = []
    for s in s0:
        n = norm(s)
        out.append(max([difflib.SequenceMatcher(None, n, t).ratio() for t in s1] or [0.0]))
    return out


SAME = """Do the two sentences below state the same information? Ignore differences in wording, order and citation markers; answer "no" only if a fact, value, name, number, date or conclusion differs.

A: {a}
B: {b}

Answer yes or no only."""


async def one_unit(llm, m, qid, u, rng, active=True):
    ins = [c for c in u["inputs"] if c in m.claims]
    if not ins:
        return []
    pick = rng.sample(ins, min(PER_LEVEL[u["level"]], len(ins)))
    if not active:   # sub-sampled unit (RNG stream is consumed identically either way)
        return []
    r = await llm.chat(prompts.CF_PERTURB.format(facts="\n".join(f"{k}|{m.claims[c]['text']}" for k, c in enumerate(pick))),
                       max_tokens=60 * len(pick) + 50)
    cf = {}
    for line in r.splitlines():
        mm = re.match(r"\s*(\d+)\s*\|\s*(.+)$", line)
        if mm and int(mm.group(1)) < len(pick):
            cf[pick[int(mm.group(1))]] = mm.group(2).strip()
    mt = MAXTOK[u["level"]]
    u0 = (await llm.chat(unit_prompt(m, u), max_tokens=mt)).strip()
    sents = split_sentences(u0)
    if not sents:
        return []
    cites = [set(re.findall(r"c\d+", u0[a:b])) for a, b in sents]

    async def cont(prompt, k):
        a, b = sents[k]
        out = await llm.complete(chat_prefix(prompt) + u0[:a], max_tokens=(b - a) // 3 + 16)
        return first_sentence(out)

    async def differs(x, f):
        if norm(x) == norm(f):
            return 0
        r = await llm.chat(SAME.format(a=strip_cites(f), b=strip_cites(x)), max_tokens=4)
        return int("no" in r.lower())

    p0 = unit_prompt(m, u)
    fact = await asyncio.gather(*(cont(p0, k) for k in range(len(sents))))
    rows = []

    async def inter(c, t):
        if norm(t) == norm(m.claims[c]["text"]):
            return
        p1 = unit_prompt(m, u, {c: t})
        cfs = await asyncio.gather(*(cont(p1, k) for k in range(len(sents))))
        y = await asyncio.gather(*(differs(x, f) for x, f in zip(cfs, fact)))
        rows.append({"qid": qid, "uid": u["id"], "level": u["level"], "cid": c, "claim": m.claims[c]["text"],
                     "cf_claim": t, "text": u0, "y": y, "lit": [int(norm(x) != norm(f)) for x, f in zip(cfs, fact)],
                     "cf_sents": cfs, "cited": [c in s for s in cites]})

    await asyncio.gather(*(inter(c, t) for c, t in cf.items()))
    return rows


async def one_user(llm, qid, out, seed):
    p = f"{STATES}/{qid}.json"
    if not os.path.exists(p):
        return 0
    m = CoCoMem.from_json(llm, json.load(open(p)))
    rng = random.Random(f"{seed}-{qid}")
    notes = [k for k, u in m.units.items() if u["level"] == "note"]
    keep = set(notes[::NOTE_STRIDE])
    res = await asyncio.gather(*(one_unit(llm, m, qid, u, rng, u["level"] != "note" or k in keep)
                                 for k, u in m.units.items()))
    rows = [r for rs in res for r in rs]
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


async def main(a):
    sp = add_test200(make_splits())
    qids = []
    for s in a.splits.split(","):
        name, _, k = s.partition(":")
        qs = sp[name]
        qids += qs[:int(k)] if k else qs
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["qid"] for l in open(a.out)}
    llm = LLM("qwen3-8b", 18431, concurrency=a.conc)
    n = await gather_limited([one_user(llm, q, a.out, a.seed) for q in qids if q not in done], desc="cfgen", every=2)
    print("rows", sum(n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--out", default="/data/yaokang/corrmem/data/cf_dev.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conc", type=int, default=128)
    asyncio.run(main(ap.parse_args()))
