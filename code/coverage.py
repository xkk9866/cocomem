"""Information coverage of repaired memories, scored without a judge.
Why: yes/no judges cannot tell a repair from a deletion (the judge calibration of q21 shows that they do not see corrected
aggregates).  Here every sentence that the reference memory C(delta(H)) (recompress) changed gives one question and its short
answer, written from the REFERENCE sentence; a reader that sees only the repaired unit answers the question ("unknown" if the
text does not state it); the answer is correct if it contains every number and name of the reference answer (string match, no
judge).  A deleted derived conclusion answers "unknown"; an unrepaired one answers with the old value.
Reported per method: coverage (share of questions answered correctly), unknown (share answered "unknown").
python coverage.py --methods append,coarse,... --port 18433"""
import argparse
import asyncio
import glob
import json
import re

from cocomem import split_sentences
from llm import LLM, gather_limited, parse_json

ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
Q = """A user corrected something they told their assistant: "{cm}"
After the correction, the assistant's memory contains this sentence:
"{sent}"
Write ONE question that this sentence answers and whose answer depends on the corrected information, and the short answer exactly as the sentence states it (a number, a name, a date or a few words).
Return only JSON: {{"question": "...", "answer": "..."}}"""
R = """Memory text:
<<<{text}>>>

Question: {q}
Answer in a few words, using only the memory text. If the text does not state the answer, answer exactly: unknown"""

sents = lambda t: [re.sub(r"\s+", " ", t[a:b]).strip() for a, b in split_sentences(t)]
norm = lambda s: " ".join(re.sub(r"[^a-z0-9. ]", " ", s.lower()).split())


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower()) if t not in {"the", "a", "an", "of", "and", "to", "in", "on", "is", "was", "are"}]


def hit(gold, ans):
    g, a = toks(gold), set(toks(ans))
    return bool(g) and all(t in a for t in g)


def load(tag, strat):
    out = {}
    for f in glob.glob(f"{ROOT}/results/{tag}/{strat}/*.json"):
        d = json.load(open(f))
        out[d.get("qid")] = d
    return out


async def main(a):
    llm = LLM("qwen3-8b", a.port, concurrency=a.conc)
    corr = {r["qid"]: r for r in json.load(open(f"{ROOT}/data/lme_corr.json"))}
    ref_all = load(a.tag, "recompress")
    items = []
    for q, ref in ref_all.items():
        if "S1" not in ref.get("units", {}) or "S1" not in ref.get("repair", {}):
            continue
        clo = ref["repair"]["S1"].get("closure", [])
        rev = ref["repair"]["S1"].get("revised", [])
        if not clo or not rev:
            continue
        st = json.load(open(f"{ROOT}/states/cocomem/{q}.json"))
        old = " ".join(st["claims"][c]["text"] for c in rev if c in st["claims"])
        for uid in clo:
            b, r = ref["units"]["S0"].get(uid), ref["units"]["S1"].get(uid)
            if b is None or r is None:
                continue
            bs = {norm(s) for s in sents(b)}
            ch = [s for s in sents(r) if norm(s) not in bs][:3]
            for k, s in enumerate(ch):
                items.append({"q": q, "uid": uid, "k": k, "sent": s, "cm": corr[q]["affect"]["correction_message"], "old": old})
    print("reference sentences", len(items), flush=True)

    async def gen(it):
        r = await llm.chat(Q.format(cm=it["cm"], sent=it["sent"]), max_tokens=200)
        js = parse_json(r, {})
        if isinstance(js, dict) and js.get("question") and js.get("answer"):
            it["question"], it["gold"] = str(js["question"]), str(js["answer"])
        return it
    items = [i for i in await gather_limited([gen(i) for i in items], desc="questions", every=500) if "gold" in i]
    print("questions", len(items), flush=True)
    summary = {}
    for strat in a.methods.split(","):
        base = strat == "orig"
        R_ = load(a.tag, "recompress" if base else strat)

        async def one(it):
            d = R_.get(it["q"])
            if not d or "S1" not in d.get("units", {}):
                return None
            text = d["units"]["S0" if base else "S1"].get(it["uid"])
            if text is None:
                return None
            ans = (await llm.chat(R.format(text=text[:6000], q=it["question"]), max_tokens=60)).strip()
            return {"q": it["q"], "uid": it["uid"], "ans": ans, "ok": hit(it["gold"], ans), "unknown": ans.lower().startswith("unknown")}
        res = [x for x in await gather_limited([one(i) for i in items], desc=strat, every=1000) if x]
        ok = [x["ok"] for x in res]
        qs = sorted({x["q"] for x in res})
        summary[strat] = {"n": len(res), "coverage": 100 * sum(ok) / max(1, len(ok)),
                          "unknown": 100 * sum(x["unknown"] for x in res) / max(1, len(res)),
                          "q_coverage": 100 * sum(all(x["ok"] for x in res if x["q"] == q) for q in qs) / max(1, len(qs)),
                          "questions": len(qs)}
        print(strat, summary[strat], flush=True)
        json.dump(res, open(f"{ROOT}/results/cov_{a.tag}_{strat}.json", "w"))
    out = f"{ROOT}/results/coverage_{a.tag}{a.suffix}.json"
    try:
        old = json.load(open(out))
    except Exception:
        old = {}
    old.update(summary)
    json.dump({"summary": old, "items": items}, open(out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main3")
    ap.add_argument("--methods", default="orig,append,slots,simupdate,self_span,coarse,ncf,spec,comm_zs,comm10,comm11,comm12,comm13,comm13L,rebuild,recompress,del_oracle,del_comm10")
    ap.add_argument("--port", type=int, default=18433)
    ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--suffix", default="")
    asyncio.run(main(ap.parse_args()))
