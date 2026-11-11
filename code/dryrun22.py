"""Dry run of the v22 correction path without a model server (CPU only): every strategy of the v22 evaluation is run
on a few record-first memories with a stand-in for the models.  Calls that are in the response cache are replayed
(the extraction of the correction sessions); every other call gets a well-formed canned output and is NOT written to
the cache.  This checks the control flow (intake, version slots, closure, record anchoring, views-only reading,
serialisation), not accuracy.
  CM_RUN_ROOT=/data/yaokang/corrmem/v22 CM_COMPRESSOR=plan python dryrun22.py [n memories] [strategies]"""
import asyncio
import collections
import hashlib
import json
import os
import re
import sys

os.environ.setdefault("CM_RUN_ROOT", "/data/yaokang/corrmem/v22")
os.environ.setdefault("CM_COMPRESSOR", "plan")
os.environ["CM_EMB_DEVICE"] = "cpu"
os.environ["CM_KV_GATE"] = "0"
import cocomem                                              # noqa: E402
from cocomem import CoCoMem, STRATEGIES, reader_answer      # noqa: E402
from data import load_lme                                   # noqa: E402
from llm import LLM, Usage                                  # noqa: E402
from scenarios import corr_session, load_corr, queries      # noqa: E402
from splits import add_test200, make_splits                 # noqa: E402

ROOT = os.environ["CM_RUN_ROOT"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STRATS = (sys.argv[2] if len(sys.argv) > 2 else "comm22,comm22_sel,comm22_overwrite,comm22_blind,comm22_nd,comm22_w,slots,recompress").split(",")
_FACT = re.compile(r"^\[(c\d+)\] \(([^)]*)\) (.*)$", re.M)


def canned(p):
    """A well-formed output for a prompt the cache does not hold."""
    if '"now"' in p and "Stored facts" in p:                          # resolve: revise the first candidate
        m = _FACT.search(p.split("Stored facts", 1)[1])
        return "resolve", json.dumps({"revised": [{"id": m.group(1), "now": m.group(3).rstrip(".") + ", as corrected today."}]}) if m else '{"revised": []}'
    if p.startswith("A memory sentence was derived from facts") or "Derivation record:" in p:
        rec = re.search(r"^Derivation record: (.*)$", p, re.M)
        sent = re.search(r"^Sentence: (.*)$", p, re.M)
        old = re.search(r"^Correction: \[(c\d+)\]", p, re.M)
        r, s = (rec.group(1) if rec else "none"), (sent.group(1) if sent else "")
        hit = old and re.search(rf"\b{old.group(1)}\b", r + " " + s)
        return "commutator", f"Derivation: {r}{' (redone)' if hit else ''}\nSentence: {s}{' (as corrected)' if hit else ''}"
    if "Correction:" in p and re.search(r"^Sentence: ", p, re.M):     # two-input and pointer operators: one line
        sent = re.search(r"^Sentence: (.*)$", p, re.M).group(1)
        return "operator", sent + " (as corrected)"
    if re.search(r"^\s*T\s*:", p, re.M) and re.search(r"^\s*D\s*:", p, re.M) and re.search(r"^\s*S\s*:", p, re.M):
        ids = [m.group(1) for m in _FACT.finditer(p)][-3:]             # record-first unit over its last facts
        return "plan", "\n".join(f"T: fact {c}\nD: {c} = its value\nS: A statement of fact {c} [{c}]." for c in ids) or "T: x\nD: none\nS: Nothing."
    if '"revised"' in p:
        return "ground", '{"revised": []}'
    if "Answer" in p and "question" in p.lower():
        return "reader", "dry-run answer"
    return "other", "A dry-run sentence."


class Fake(LLM):
    def __init__(self, model, **kw):
        super().__init__(model, 1, **kw)
        self.miss, self.hits = collections.Counter(), 0

    async def chat(self, messages, max_tokens=1024, temperature=0.0, usage=None, think=None, stop=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        think = self.think if think is None else think
        key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, temperature, think, stop]).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            if usage is not None:
                usage.add(hit["p"], hit["c"])
            return hit["text"]
        kind, text = canned(messages[-1]["content"])
        self.miss[kind] += 1
        if usage is not None:
            usage.add(len(messages[-1]["content"]) // 4, len(text) // 4)
        return text

    async def complete(self, prompt, max_tokens=64, usage=None):
        self.miss["complete"] += 1
        return " yes"


async def main():
    items = {x["qid"]: x for x in load_lme()}
    corr = load_corr()
    sp = add_test200(make_splits())
    qids = []
    for q in sp["test200"]:
        p = f"{ROOT}/states/cocomem/{q}.json"
        if q in corr and "affect" in corr[q] and os.path.exists(p):
            d = json.load(open(p))
            if d.get("units") and (cocomem.COMPRESSOR != "plan" or all(u.get("plan_ok") is not None for u in d["units"].values())) \
                    and not any(u["level"] == "profile" for u in d["units"].values()):
                qids.append(q)
        if len(qids) >= N:
            break
    print(f"memories: {qids}; compressor={cocomem.COMPRESSOR} intake={cocomem.INTAKE} profile level={cocomem.PROFILE_LEVEL}")
    llm, tnet = Fake("qwen3-8b"), Fake("tnet22", cache_name="dryrun_tmp")
    for strategy in STRATS:
        cfg = STRATEGIES[strategy]
        for q in qids:
            inst, rec = items[q], corr[q]
            base = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{q}.json")))
            if cfg.get("lineage"):
                base.set_lineage(cfg["lineage"])
            if cfg.get("tnet") == "student":
                base.tnet = tnet
            if cfg.get("deriv"):
                from deriv import attach as attach_deriv
                attach_deriv(base, f"{ROOT}/states/deriv/{q}.json")
            m = base.clone()
            u = Usage()
            info = await m.correct(corr_session(inst, rec, "affect"), strategy=strategy, usage=u)
            changed = [uid for uid, x in m.units.items() if uid in base.units and x["text"] != base.units[uid]["text"]]
            added = [uid for uid in m.units if uid not in base.units]
            slots = [(c, x.get("since"), x["date"][:10]) for c, x in m.claims.items() if x.get("revises")]
            bad = [uid for uid, x in m.units.items() if any(c not in m.claims for c in x["inputs"])] if cfg.get("history", True) else []
            stale_in = [uid for uid, x in m.units.items() if set(x["inputs"]) & set(info.get("revised", []))]
            ans = {}
            for sc, qid, question, gold, jt in queries(inst, rec):
                if sc == "S1" and qid in ("q", "hist"):
                    for lev in (None, ("note", "digest")):
                        chosen, ctx = await m.retrieve(question, budget=4000, levels=lev)
                        hyp, _ = await reader_answer(llm, ctx, question, inst["qdate"], Usage())
                        ans[(qid, "views" if lev else "all")] = (len(chosen), len(ctx) // 4, sorted({t[0] for t in chosen}))
            json.dumps(m.to_json())
            print(f"{strategy:17s} {q} edits={len(info.get('edits', {}))} revised={info.get('revised')} closure={info.get('closure')} "
                  f"repaired={info.get('repaired')} mapped={info.get('mapped_sents')}/{info.get('closure_sents')} changed={changed} new units={added} "
                  f"slots={slots} dangling={bad} units still on the old claim={stale_in} read={ans} usage={u.as_dict()}", flush=True)
            for n, (o, ot, nt) in list(info.get("edits", {}).items())[:1]:
                print(f"      edit {o} -> {n}: {ot[:90]!r} => {nt[:90]!r}")
                cl = m.claims[n]
                print("      slot as read:", m.slot_text(n)[:260])
    print("8B stand-in: cache hits", llm.hits, "canned", dict(llm.miss), "| student stand-in: canned", dict(tnet.miss))


asyncio.run(main())
