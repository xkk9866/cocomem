"""Build LongMemEval-Corr: for each question, an evidence-targeting correction (answer changes),
an unrelated correction with a probe question (answer of the main question unchanged),
and a history question. Generated with Qwen3-14B-AWQ (thinking mode); verified later."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re

from data import load_lme
from llm import LLM, gather_limited, parse_json
from splits import add_test200, make_splits

OUT = "/data/yaokang/corrmem/data/lme_corr.json"

AFFECT = """You are helping build a benchmark that tests whether a personal assistant's long-term memory handles user corrections.

The user had many chat sessions with an assistant. In the sessions below (the "evidence"), the user gave information that answers a later question.

Question type: {qtype}
Question (asked on {qdate}): {question}
Correct answer (before any correction): {answer}

Evidence turns:
{evidence}

Task: write ONE short message that the user sends in a NEW chat session (after all the sessions above, before the question is asked) in which the user explicitly corrects one specific piece of information they gave earlier, so that the correct answer to the question changes.
Requirements:
- The correction must be a realistic slip the user is fixing (a wrong number, name, date, place, item, or an event that did not actually happen as described). It must clearly state both what was said before and the corrected information, e.g. "Oh, I need to correct something I told you earlier: ...".
- Do NOT mention or hint at the question itself. Do not change anything else.
- For counting/aggregation questions, correct one counted item (e.g. its quantity, or that one item did not actually happen); for time questions, correct the date or timing of one event; the new answer must follow unambiguously from the evidence plus the correction.
- The new answer must be different from the original answer.
- Also write a history question asking what the user ORIGINALLY said about the corrected detail (before correcting it), and its answer.
- Also write a short, natural assistant reply acknowledging the correction (do not repeat the question).

Return only JSON:
{{"corrected_detail": "...", "correction_message": "...", "assistant_reply": "...", "new_answer": "...", "history_question": "...", "history_answer": "..."}}"""

UNREL = """You are helping build a benchmark that tests whether a personal assistant's long-term memory handles user corrections.

Below are the user's messages from one earlier chat session (dated {date}).

{turns}

Task: pick ONE specific, concrete personal detail the user stated in this session (a number, name, place, date, item, or event about the user's own life). Write a short message the user sends in a NEW chat session in which the user explicitly corrects that detail (e.g. "By the way, I need to correct something I mentioned before: ..."), stating both the old and the new information. Then write a question that can only be answered with the corrected detail, its answer (the new value), and a short assistant reply acknowledging the correction.
If the session contains no concrete personal detail of the user, return {{"none": true}}.

Return only JSON:
{{"corrected_detail": "...", "correction_message": "...", "assistant_reply": "...", "probe_question": "...", "probe_answer": "...", "old_value": "..."}}"""


def evidence_text(inst, max_chars=1500):
    blocks = []
    for s in inst["sessions"]:
        if not s["evidence"]:
            continue
        turns = [(i, t) for i, t in enumerate(s["turns"]) if t.get("has_answer")]
        if not turns:  # fall back to user turns of the evidence session
            turns = [(i, t) for i, t in enumerate(s["turns"]) if t["role"] == "user"][:4]
        lines = [f"[Session {s['date']}]"]
        for i, t in turns:
            c = t["content"].strip()
            lines.append(f"{'User' if t['role'] == 'user' else 'Assistant'}: {c[:max_chars]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def personal(turns):
    txt = " ".join(t["content"] for t in turns if t["role"] == "user")
    return len(re.findall(r"\b(I|my|My|I'm|I've|we|our)\b", txt)) >= 4 and len(txt) < 6000


async def gen_one(llm, inst, rng):
    rec = {"qid": inst["qid"], "qtype": inst["qtype"]}
    if inst["qtype"] != "single-session-assistant":
        r = await llm.chat(AFFECT.format(qtype=inst["qtype"], qdate=inst["qdate"], question=inst["question"],
                                         answer=inst["answer"], evidence=evidence_text(inst)),
                           max_tokens=6000, temperature=0.6)
        js = parse_json(r, None)
        if isinstance(js, dict) and js.get("correction_message") and js.get("new_answer"):
            rec["affect"] = js
    cands = [s for s in inst["sessions"] if not s["evidence"] and personal(s["turns"])]
    rng.shuffle(cands)
    for s in cands[:3]:
        turns = "\n".join(f"User: {t['content'][:800]}" for t in s["turns"] if t["role"] == "user")
        r = await llm.chat(UNREL.format(date=s["date"], turns=turns), max_tokens=4000, temperature=0.6)
        js = parse_json(r, None)
        if isinstance(js, dict) and not js.get("none") and js.get("correction_message") and js.get("probe_answer"):
            js["sid"] = s["sid"]
            rec["unrel"] = js
            break
    return rec


async def main(args):
    sp = add_test200(make_splits())
    qids = set(sp["dev"] + sp["test200"])
    items = [x for x in load_lme() if x["qid"] in qids]
    if args.limit:
        items = items[:args.limit]
    done = {}
    if os.path.exists(OUT):
        done = {r["qid"]: r for r in json.load(open(OUT))}
    todo = [x for x in items if x["qid"] not in done]
    llm = LLM("qwen3-8b", 18431, concurrency=64, think=True)
    recs = await gather_limited([gen_one(llm, x, random.Random(int(hashlib.md5(x["qid"].encode()).hexdigest()[:8], 16))) for x in todo],
                                desc="gencorr", every=20)
    for r in recs:
        done[r["qid"]] = r
    json.dump(list(done.values()), open(OUT, "w"), indent=1)
    print("affect", sum("affect" in r for r in done.values()), "unrel", sum("unrel" in r for r in done.values()),
          "total", len(done))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
