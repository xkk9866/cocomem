"""LoCoMo-Corr: multiple sequential corrections per conversation on LoCoMo (ACL'24).

For each of the 10 conversations, K questions (single-hop / multi-hop / temporal) are selected with
non-overlapping evidence; for each, the speaker who stated the evidence later sends a correction.
All K corrections arrive one after another at the end of the conversation (interference test).
Queries: affected questions (new answers), history questions (old values), unaffected questions
(original answers, evidence in sessions untouched by any correction)."""
import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import re
import time

from cocomem import CoCoMem, reader_answer, STRATEGIES
from llm import LLM, Usage, gather_limited, parse_json

ROOT = __import__("os").environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
CORR = f"{ROOT}/data/locomo_corr.json"
CAT = {1: "multi-hop", 2: "temporal", 4: "single-hop", 3: "open-domain", 5: "adversarial"}


def parse_date(s):
    return dt.datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def fmt(d):
    return d.strftime("%Y/%m/%d (%a) %H:%M")


def load_locomo():
    data = json.load(open(f"{ROOT}/data/locomo10.json"))
    convs = []
    for x in data:
        c = x["conversation"]
        a, b = c["speaker_a"], c["speaker_b"]
        sessions = []
        k = 1
        while f"session_{k}" in c:
            turns = []
            for t in c[f"session_{k}"]:
                txt = t["text"]
                if t.get("blip_caption"):
                    txt += f" [shares a photo: {t['blip_caption']}]"
                turns.append({"role": t["speaker"], "content": txt, "dia_id": t["dia_id"]})
            sessions.append({"sid": f"S{k}", "date": fmt(parse_date(c[f"session_{k}_date_time"])), "turns": turns,
                             "evidence": False})
            k += 1
        dia = {t["dia_id"]: (s["sid"], t) for s in sessions for t in s["turns"]}
        convs.append({"cid": x["sample_id"], "persons": [a, b], "sessions": sessions, "qa": x["qa"], "dia": dia})
    return convs


GEN = """Two friends, {a} and {b}, talk regularly. Below is a question about their conversations, its correct answer, and the conversation turns that contain the answer.

Question ({cat}): {question}
Correct answer: {answer}

Evidence turns:
{evidence}

Task: write ONE short message that {speaker} sends to {other} in a LATER conversation, explicitly correcting one specific piece of information {speaker} gave in the evidence (a wrong number, name, date, place, item, or an event that did not happen as described), so that the correct answer to the question changes.
Requirements:
- It must clearly state both what was said before and the corrected information, e.g. "Oh, I have to correct something I told you: ...".
- Do NOT mention the question. Change nothing else. The new answer must follow unambiguously from the evidence plus the correction, and must differ from the original answer.
- Also write a short reply from {other}, a history question asking what {speaker} ORIGINALLY said about the corrected detail, and its answer.

Return only JSON:
{{"correction_message": "...", "reply": "...", "new_answer": "...", "history_question": "...", "history_answer": "..."}}"""


async def gen(args):
    convs = load_locomo()
    llm = LLM("qwen3-8b", 18431, concurrency=64, think=True)
    out = {}
    jobs = []
    for cv in convs:
        rng = random.Random(int(hashlib.md5(cv["cid"].encode()).hexdigest()[:8], 16))
        qa = [q for q in cv["qa"] if q.get("category") in (1, 2, 4) and q.get("evidence") and "answer" in q]
        rng.shuffle(qa)
        used_sess, picked = set(), []
        for q in qa:
            ev = [cv["dia"].get(e) for e in q["evidence"]]
            if not ev or any(v is None for v in ev):
                continue
            sess = {v[0] for v in ev}
            spk = {v[1]["role"] for v in ev}
            if len(spk) != 1 or sess & used_sess:
                continue
            picked.append(q)
            used_sess |= sess
            if len(picked) >= args.k:
                break
        unaff = [q for q in qa if q not in picked and
                 all(cv["dia"].get(e) and cv["dia"][e][0] not in used_sess for e in q["evidence"])]
        rng.shuffle(unaff)
        out[cv["cid"]] = {"affect": [], "unaffected": unaff[:args.n_unaff]}
        for q in picked:
            ev = [cv["dia"][e] for e in q["evidence"]]
            speaker = ev[0][1]["role"]
            other = [p for p in cv["persons"] if p != speaker][0]
            evtxt = "\n".join(f"[{s}] {t['role']}: {t['content']}" for s, t in ev)
            jobs.append((cv["cid"], q, speaker, other, llm.chat(GEN.format(
                a=cv["persons"][0], b=cv["persons"][1], cat=CAT[q["category"]], question=q["question"],
                answer=q["answer"], evidence=evtxt, speaker=speaker, other=other), max_tokens=6000, temperature=0.6)))
    res = await gather_limited([j[4] for j in jobs], desc="gen-locomo", every=20)
    for (cid, q, spk, oth, _), r in zip(jobs, res):
        js = parse_json(r, None)
        if isinstance(js, dict) and js.get("correction_message") and js.get("new_answer"):
            out[cid]["affect"].append({"qa": q, "speaker": spk, "other": oth, **js})
    json.dump(out, open(CORR, "w"), indent=1)
    print({k: (len(v["affect"]), len(v["unaffected"])) for k, v in out.items()})


def corr_sessions(cv, rec):
    last = max(dt.datetime.strptime(s["date"][:10] + s["date"][-6:], "%Y/%m/%d %H:%M") for s in cv["sessions"])
    out = []
    for i, a in enumerate(rec["affect"]):
        d = last + dt.timedelta(days=i + 1)
        out.append({"sid": f"corr_{i}", "date": fmt(d), "evidence": False,
                    "turns": [{"role": a["speaker"], "content": a["correction_message"]},
                              {"role": a["other"], "content": a.get("reply") or "Thanks for telling me!"}]})
    return out


def query_list(rec):
    qs = []
    for i, a in enumerate(rec["affect"]):
        q = a["qa"]
        cat = CAT[q["category"]]
        qs.append(("S0", f"a{i}", q["question"], str(q["answer"]), "temporal-reasoning" if cat == "temporal" else "multi-session", cat))
        qs.append(("S1", f"a{i}", q["question"], str(a["new_answer"]), "temporal-corrected" if cat == "temporal" else "knowledge-update", cat))
        qs.append(("S1", f"h{i}", a["history_question"], str(a["history_answer"]), "history", cat))
    for i, q in enumerate(rec["unaffected"]):
        cat = CAT[q["category"]]
        jt = "temporal-reasoning" if cat == "temporal" else "multi-session"
        qs.append(("S0", f"u{i}", q["question"], str(q["answer"]), jt, cat))
        qs.append(("S1", f"u{i}", q["question"], str(q["answer"]), jt, cat))
    return qs


async def run(args):
    convs = {c["cid"]: c for c in load_locomo()}
    corr = json.load(open(CORR))
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=128)
    from baselines import SYSTEMS

    async def one(method, cv):
        out = f"{ROOT}/results/{'locomo_reg' if args.regress else 'locomo'}/{method}/{cv['cid']}.json"
        if os.path.exists(out):
            return "cached"
        rec = corr[cv["cid"]]
        sp = f"{ROOT}/states/locomo_{'cocomem' if method in STRATEGIES or method in ('append', 'simupdate') else method}/{cv['cid']}.json"
        is_ours = method in STRATEGIES or method in ("append", "simupdate")
        if os.path.exists(sp):
            st = json.load(open(sp))
            m = CoCoMem.from_json(llm, st) if is_ours else SYSTEMS[method].from_json(llm, st)
        else:
            t0 = time.time()
            if is_ours:
                m = CoCoMem(llm, persons=cv["persons"])
                raws = await asyncio.gather(*(m.extract(s, m.usage["write"]) for s in cv["sessions"]))
                notes = []
                for s, raw in zip(cv["sessions"], raws):
                    new = []
                    for r in raw:
                        cid = m._new_cid()
                        m.claims[cid] = {"id": cid, "text": r["text"], "speaker": r["speaker"], "turn": r["turn"],
                                         "sid": s["sid"], "date": s["date"], "status": "current", "superseded_by": [],
                                         "superseded_at": None, "revises": [], "revises_hint": r["revises"]}
                        new.append(cid)
                    if new:
                        notes.append(m._make_note(s, new, m.usage["write"]))
                await asyncio.gather(*notes)
                await m.consolidate()
            else:
                m = SYSTEMS[method](llm)
                await m.build(cv["sessions"])
            st = m.to_json()
            st["build_time"] = time.time() - t0
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            json.dump(st, open(sp, "w"))
        if is_ours:
            lin = STRATEGIES.get(method, {}).get("lineage")
            if lin == "ncf":
                from run_corr import get_clf
                m.set_lineage("ncf", cle=get_clf(args.clf))
            elif lin:
                m.set_lineage(lin)
            if STRATEGIES.get(method, {}).get("repair") == "ccf":
                from run_corr import get_clf
                m.clf = get_clf(args.clf)
            if STRATEGIES.get(method, {}).get("tnet") == "student":
                from run_corr import get_tnet
                m.tnet = get_tnet(STRATEGIES[method].get("tnet_name", "tnet"))
            if STRATEGIES.get(method, {}).get("witness") == "annot":
                from witness import attach
                attach(m, f"{ROOT}/states/witness_locomo/{cv['cid']}.json")
            if STRATEGIES.get(method, {}).get("deriv"):
                from deriv import attach as attach_deriv
                attach_deriv(m, f"{ROOT}/states/deriv_locomo/{cv['cid']}.json")
        base = m.clone()
        mem = m.clone()
        rep = []
        u = Usage()
        t0 = time.time()
        reg = []
        for step, s in enumerate(corr_sessions(cv, rec)):        # sequential corrections
            u0 = u.as_dict()
            if is_ours:
                info = await mem.correct(s, strategy=method, usage=u)
            else:
                info = await mem.correct(s, usage=u)
            u1 = u.as_dict()
            info["usage"] = {k_: u1[k_] - u0[k_] for k_ in u1}
            rep.append(info)
            if args.regress:     # after EVERY correction: re-ask all corrections made so far and a fixed unaffected panel
                reg.append((step, mem.clone()))
        res = {"cid": cv["cid"], "method": method, "repair": {"S1": {"usage": u.as_dict(), "time": time.time() - t0, "steps": rep}},
               "answers": []}
        persons = cv["persons"]
        if args.regress:
            async def rans(step, mm, key, q, gold, jt, cat, kind):
                _, ctx = await mm.retrieve(q, budget=args.budget)
                hyp, full = await reader_answer(llm, ctx, q, "", Usage(), persons=persons)
                res["answers"].append({"scenario": f"R{step}", "qkey": key, "question": q, "gold": gold, "jtype": jt,
                                       "cat": cat, "hyp": hyp, "full": full, "step": step, "kind": kind})
            jobs = []
            qs = query_list(rec)
            for step, mm in reg:
                for sc, key, q, gold, jt, cat in qs:
                    if sc != "S1":
                        continue
                    if key.startswith("a") and int(key[1:]) <= step:
                        jobs.append(rans(step, mm, key, q, gold, jt, cat, "fresh" if int(key[1:]) == step else "carried"))
                    elif key.startswith("u") and int(key[1:]) < 5:
                        jobs.append(rans(step, mm, key, q, gold, jt, cat, "unaffected"))
            await asyncio.gather(*jobs)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(res, open(out, "w"))
            return "ok"

        async def ans(sc, key, q, gold, jt, cat):
            mm = base if sc == "S0" else mem
            _, ctx = await mm.retrieve(q, budget=args.budget)
            hyp, full = await reader_answer(llm, ctx, q, "", Usage(), persons=persons)
            res["answers"].append({"scenario": sc, "qkey": key, "question": q, "gold": gold, "jtype": jt, "cat": cat,
                                   "hyp": hyp, "full": full})
        await asyncio.gather(*(ans(*x) for x in query_list(rec)))
        res["mem_size"] = base.size_tokens()
        if is_ours:
            res["units"] = {"S0": {k: x["text"] for k, x in base.units.items()},
                            "S1": {k: x["text"] for k, x in mem.units.items()}}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"))
        return "ok"

    for method in args.methods.split(","):
        r = await gather_limited([one(method, cv) for cv in (list(convs.values())[::-1] if os.environ.get("CM_REVERSE") else convs.values())], desc=f"locomo-{method}", every=5)
        print(method, r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n_unaff", type=int, default=40)
    ap.add_argument("--methods", default="append")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--clf", default="/data/yaokang/corrmem/models/ncf")
    a = ap.parse_args()
    asyncio.run(gen(a) if a.cmd == "gen" else run(a))
