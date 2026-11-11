"""Intake of correction sessions on memories outside the test split (train / dev users): the two-step intake (claims
from the session alone, then verification against the ledger) next to the ledger-conditioned resolution
(CoCoMem.resolve).  The memories of the test users are never read here.
  python resolve_check.py PORT [limit] [conc]        ->  logs/resolve_check.json + a readable listing on stdout"""
import asyncio
import collections
import json
import os
import re
import sys

os.environ.pop("CM_COMPRESSOR", None)
os.environ["CM_EMB_DEVICE"] = "cpu"
from cocomem import CoCoMem                      # noqa: E402
from data import load_lme                        # noqa: E402
from llm import LLM, Usage                       # noqa: E402
from scenarios import corr_session, load_corr    # noqa: E402
from splits import add_test200, make_splits      # noqa: E402

SRC = os.environ.get("CM_ROOT", "/data/yaokang/corrmem")
PORT = int(sys.argv[1])
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
DIRECT = ("single-session-user", "single-session-assistant", "knowledge-update")


def norm(s):
    return re.sub(r"[^a-z0-9$%:. ]", " ", str(s).lower())


def has(ans, texts):
    if ans is None:
        return False
    a = norm(ans).strip()
    toks = [t for t in a.split() if re.search(r"\d", t)] or a.split()[:3]
    return bool(toks) and any(all(t in norm(x) for t in toks) for x in texts)


async def main():
    items = {x["qid"]: x for x in load_lme()}
    corr = load_corr()
    sp = add_test200(make_splits())
    test = set(sp["test200"]) | set(sp.get("test", []))
    qids = sorted(q for q in corr if q not in test and os.path.exists(f"{SRC}/states/cocomem/{q}.json"))
    if LIMIT:
        qids = qids[:LIMIT]
    print("memories outside the test split:", len(qids), flush=True)
    if PORT:
        llm = LLM("qwen3-8b", PORT, concurrency=CONC)
    else:                                                # PORT 0: the responses of an earlier run, read from the cache (CPU)
        import hashlib

        class Replay(LLM):
            missing = 0

            async def chat(self, messages, max_tokens=1024, temperature=0.0, usage=None, think=None, stop=None):
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                think = self.think if think is None else think
                key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, temperature, think, stop]).encode()).hexdigest()
                hit = self.cache.get(key)
                if hit is None:
                    Replay.missing += 1
                    return ""
                if usage is not None:
                    usage.add(hit["p"], hit["c"])
                return hit["text"]

        os.environ["CM_KV_GATE"] = "0"
        llm = Replay("qwen3-8b", 1, concurrency=CONC)
    tab = collections.Counter()
    rows = []
    sem = asyncio.Semaphore(CONC)

    async def one(q, kind):
        inst, rec = items[q], corr[q]
        if kind not in rec:
            return
        d = json.load(open(f"{SRC}/states/cocomem/{q}.json"))
        s = corr_session(inst, rec, kind)
        newv = rec[kind].get("probe_answer") if kind == "unrel" else rec[kind].get("new_answer")
        oldv = rec[kind].get("old_value") if kind == "unrel" else None
        async with sem:
            m = CoCoMem.from_json(llm, d)                    # two-step intake
            u = Usage()
            new = await m.add_session(s, usage=u, make_note=False)
            ctx = "\n".join(t["content"] for t in s["turns"] if t["role"] == "user")
            rev = await m.ground(new, s["date"], u, context=ctx)
            g_new = [m.claims[c]["text"] for c in rev]
            g_old = [m.claims[o]["text"] for c in rev for o in rev[c]]
            g_sid = [m.claims[o]["sid"] for c in rev for o in rev[c]]
            m2 = CoCoMem.from_json(llm, d)                   # ledger-conditioned resolution
            u2 = Usage()
            new2 = await m2.add_session(s, usage=u2, make_note=False)
            rev2 = await m2.resolve(s, new2, u2)
            r_new = [m2.claims[c]["text"] for c in rev2]
            r_old = [m2.claims[o]["text"] for c in rev2 for o in rev2[c]]
            r_sid = [m2.claims[o]["sid"] for c in rev2 for o in rev2[c]]
        tab[(kind, "n")] += 1
        for nm, got, new_t, old_t, sids, use in (("two-step", rev, g_new, g_old, g_sid, u), ("resolve", rev2, r_new, r_old, r_sid, u2)):
            tab[(kind, f"{nm}: corrections with a revised slot")] += bool(got)
            tab[(kind, f"{nm}: revised slots")] += len(old_t)
            tab[(kind, f"{nm}: 8B tokens")] += use.prompt_tokens + use.completion_tokens
            if kind == "unrel":
                sid = rec[kind].get("sid")
                tab[(kind, f"{nm}: new value in the new version")] += has(newv, new_t)
                tab[(kind, f"{nm}: old value in the revised claim")] += has(oldv, old_t)
                tab[(kind, f"{nm}: a revised claim from the corrected session")] += sid in sids
                tab[(kind, f"{nm}: revised claims from other sessions")] += sum(x != sid for x in sids)
            elif inst["qtype"] in DIRECT:
                tab[(kind, "n direct")] += nm == "resolve"
                tab[(kind, f"{nm}: corrected answer in the new version (direct questions)")] += has(newv, new_t)
        rows.append({"qid": q, "kind": kind, "qtype": inst["qtype"], "msg": " / ".join(t["content"] for t in s["turns"]),
                     "old_value": oldv, "new_value": newv,
                     "two_step": [[o, n] for o, n in zip(g_old, g_new)], "resolve": [[o, n] for o, n in zip(r_old, r_new)],
                     "extracted": [m2.claims[c]["text"] for c in new2 if c not in rev2]})

    await asyncio.gather(*(one(q, k) for q in qids for k in ("affect", "unrel")))
    if not PORT:
        print("replayed from the cache; calls not in the cache:", llm.missing)
    for k in sorted(tab):
        print(k, tab[k])
    rows.sort(key=lambda r: (r["kind"], r["qid"]))
    json.dump({"table": {f"{a} | {b}": v for (a, b), v in tab.items()}, "rows": rows},
              open(f"{SRC}/logs/resolve_check{'' if PORT else '_replay'}.json", "w"), indent=1)
    for r in rows:
        print("-" * 100)
        print(r["qid"], r["kind"], r["qtype"], "| new:", str(r["new_value"])[:60], "| old:", str(r["old_value"])[:60])
        print("   said:", r["msg"][:420])
        print("   extracted:", " ## ".join(r["extracted"])[:300])
        for o, n in r["two_step"][:3]:
            print("   TWO-STEP:", o[:160], "  ==>  ", n[:160])
        for o, n in r["resolve"][:5]:
            print("   RESOLVE :", o[:160], "  ==>  ", n[:160])


asyncio.run(main())
