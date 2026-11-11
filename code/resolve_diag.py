"""Why a correction leaves the ledger untouched (CPU only; memories outside the test split; responses replayed from the
cache of the resolve_check job, nothing is written).  For every correction of the listing where the ledger-conditioned
intake revised no slot: the candidate slots shown to the compressor, whether the slot the two-step intake revised was
among them, and the raw response.
  python resolve_diag.py [qid ...]"""
import asyncio
import hashlib
import json
import os
import sys

os.environ.pop("CM_COMPRESSOR", None)
os.environ["CM_EMB_DEVICE"] = "cpu"
os.environ["CM_KV_GATE"] = "0"
import cocomem                                   # noqa: E402
from cocomem import CoCoMem                      # noqa: E402
from data import load_lme                        # noqa: E402
from llm import LLM, Usage                       # noqa: E402
from scenarios import corr_session, load_corr    # noqa: E402
from splits import add_test200, make_splits      # noqa: E402

SRC = os.environ.get("CM_ROOT", "/data/yaokang/corrmem")


class Replay(LLM):
    def __init__(self, model, **kw):
        super().__init__(model, 1, **kw)
        self.log = []

    async def chat(self, messages, max_tokens=1024, temperature=0.0, usage=None, think=None, stop=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        think = self.think if think is None else think
        key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, temperature, think, stop]).encode()).hexdigest()
        hit = self.cache.get(key)
        text = hit["text"] if hit is not None else "<<NOT IN CACHE>>"
        self.log.append((messages[-1]["content"], text))
        return text


async def main():
    items = {x["qid"]: x for x in load_lme()}
    corr = load_corr()
    sp = add_test200(make_splits())
    test = set(sp["test200"]) | set(sp.get("test", []))
    rows = json.load(open(f"{SRC}/logs/resolve_check.json"))["rows"]
    want = set(sys.argv[1:])
    todo = [r for r in rows if (r["qid"] in want if want else (r["kind"] == "affect" and not r["resolve"]))]
    for r in todo:
        q, kind = r["qid"], r["kind"]
        assert q not in test
        llm = Replay("qwen3-8b")
        m = CoCoMem.from_json(llm, json.load(open(f"{SRC}/states/cocomem/{q}.json")))
        s = corr_session(items[q], corr[q], kind)
        new = await m.add_session(s, usage=Usage(), make_note=False)
        llm.log.clear()
        rev = await m.resolve(s, new, Usage())
        print("=" * 110)
        print(q, kind, items[q]["qtype"], "| said:", r["msg"][:300])
        for p, text in llm.log:
            cands = p.split("Stored facts (id, date recorded, speaker):", 1)[-1].split("\n\nDoes the new conversation", 1)[0].strip()
            two = [o for o, _ in r["two_step"]]
            print("  candidates:", cands.count("\n") + 1, "| slot of the two-step intake among them:",
                  [any(o[:80] in line for line in cands.splitlines()) for o in two])
            for line in cands.splitlines():
                mark = "*" if any(o[:80] in line for o in two) else " "
                print("   ", mark, line[:230])
            print("  response:", text[:700].replace("\n", " | "))
        print("  parsed:", {c: v for c, v in rev.items()})


asyncio.run(main())
