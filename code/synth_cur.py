"""Counterfactual aggregate curriculum (v13).

Why: on LongMemEval-Corr only a small share of the stored sentences are aggregates (totals, counts, extremes, orders), so
a student trained on those counterfactuals never practises the edits the memory needs most.  We extend the training
distribution -- not the labels' source: the compressor writes digests from LLM-generated, quantity-rich claim sets on many
themes (none of the programmatic benchmark's eight domains), a claim is corrected counterfactually, and the targets are
selected exactly as before (candidates scored by the compressor reading the corrected facts).  No program ever computes a
target; the benchmark's generator, templates and domains are not used.
Stage 1 (this file): themes -> claim sets -> digests -> counterfactuals -> candidate rewrites; writes
   models/synth/train.jsonl (relabel_cv2 input) and states/cocomem/syn_*.json (the synthetic memories).
Stage 2: commutator.py relabel_cv2 --src models/synth --out models/synth_out  (reasoning-mode repair + verification)."""
import argparse
import asyncio
import json
import os
import random
import re

from cocomem import CoCoMem, ledger_call
from commutator import T_PROMPT, norm, split
from llm import LLM, Usage, gather_limited

ROOT = "/data/yaokang/corrmem"
BENCH = ["running", "groceries", "reading pages", "overtime", "swimming", "walking steps", "piano practice", "driving"]
THEMES = """List {n} different everyday themes in which a person keeps numbers over several days or weeks (spending, sports, work, study, cooking, travel, hobbies, health, household, pets, garden, finance, media, ...). Avoid these themes: {bench}. Output one theme per line, no numbering, at most 6 words each."""
CLAIMS = """Write {n} short first-person statements, each from a different day of one month, that a person told their assistant about "{theme}". Every statement must contain one specific number (an amount, count, duration, distance, score or price) and a date, and the numbers must differ. Output one statement per line in the form: YYYY/MM/DD | statement"""
CF = """Fact: {fact}
Rewrite this fact with its key number changed to a different plausible value (do not change anything else). Output only the rewritten fact."""


def parse_lines(t):
    out = []
    for l in t.splitlines():
        m = re.match(r"\s*(\d{4}/\d{2}/\d{2})\s*\|\s*(.+)", l)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


async def main(a):
    rnd = random.Random(a.seed)
    llm = LLM("qwen3-8b", 18431, concurrency=160)
    usage = Usage()
    raw = await llm.chat(THEMES.format(n=a.themes, bench=", ".join(BENCH)), max_tokens=2500)
    themes = [re.sub(r"^[\d\.\-\*\s]+", "", t).strip() for t in raw.splitlines() if t.strip()][:a.themes]
    print("themes", len(themes), flush=True)
    jobs = [(i, themes[i % len(themes)]) for i in range(a.users)]

    async def make(job):
        i, theme = job
        n = rnd.randint(6, 9)
        t = await llm.chat(CLAIMS.format(n=n, theme=theme) + f"\n(variant {i})", max_tokens=900, temperature=0.7)
        rows = parse_lines(t)
        if len(rows) < 5:
            return None
        m = CoCoMem(llm)
        for j, (d, txt) in enumerate(rows, 1):
            m.claims[f"c{j}"] = {"id": f"c{j}", "text": txt, "speaker": "user", "turn": 0, "sid": f"syn{i}", "date": f"{d} (Mon) 10:00",
                                 "status": "current", "superseded_by": [], "superseded_at": None, "revises": [], "revises_hint": None}
        m.n = len(rows)
        await m._make_digest("d0", [f"c{j}" for j in range(1, len(rows) + 1)], usage)
        json.dump(m.to_json(), open(f"{ROOT}/states/cocomem/syn_{i}.json", "w"))
        return i, m

    mems = [x for x in await gather_limited([make(j) for j in jobs], desc="users", every=50) if x]
    print("users", len(mems), flush=True)

    async def group(i, m, cid):
        u = m.units["d0"]
        sents = split(u["text_cited"])
        fact = m.claims[cid]["text"]
        new = (await llm.chat(CF.format(fact=fact), max_tokens=120)).strip().split("\n")[0]
        if not new or norm(new) == norm(fact):
            return []
        old = f"[{cid}] {fact}"
        scope = m._derivation_scope(u, {cid}, sents, list(range(len(sents))))
        lines = [f"[{c}] ({m.claims[c]['date'][:10]}, user) {new if c == cid else m.claims[c]['text']}" for c in scope]
        loc, led = await asyncio.gather(
            asyncio.gather(*(llm.chat(T_PROMPT.format(old=old, new=new, sent=s), max_tokens=300) for s in sents)),
            ledger_call(llm, "digest", old, new, "\n".join(lines), sents, list(range(len(sents)))))
        out = []
        for k, s in enumerate(sents):
            first = lambda x: x.strip().split("\n")[0].strip().strip('"')
            out.append({"qid": f"syn_{i}", "uid": "d0", "level": "digest", "k": k, "y": 0, "old": old, "new": new, "sent": s,
                        "target": s, "target5": first(loc[k]), "target7": None, "t_ledger": led.get(k) or s, "cited": 0})
        return out

    tasks = []
    for i, m in mems:
        cids = list(m.claims)
        rnd.shuffle(cids)
        for cid in cids[:a.per_user]:
            tasks.append(group(i, m, cid))
    G = await gather_limited(tasks, desc="groups", every=100)
    E = [e for g in G for e in g]
    os.makedirs(f"{ROOT}/models/synth", exist_ok=True)
    with open(f"{ROOT}/models/synth/train.jsonl", "w") as f:
        for e in E:
            f.write(json.dumps(e) + "\n")
    with open(f"{ROOT}/models/synth/test.jsonl", "w") as f:           # relabel_cv2 loops over train and test
        for e in E[:12]:
            f.write(json.dumps(e) + "\n")
    print("pairs", len(E), "groups", len(G), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", type=int, default=150)
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--per_user", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    asyncio.run(main(ap.parse_args()))
