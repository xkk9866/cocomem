"""v21: the derivation-conditioned commutator T_theta(delta, D_k, s_k) -> (D'_k, s'_k).

The stored sentence s_k carries a materialized derivation record D_k (deriv.py): the claims it was derived from, each
with the value the sentence took from it, and the computation.  The operator reads the correction, the record and the
sentence -- nothing else, no ledger -- and writes the record FIRST (the delta rule made explicit: the corrected operand
replaces the old one, the computation is redone) and then the sentence.  Copies are exact.

Supervision closes the commutation square as in v20: for every counterfactual correction of a training unit, candidate
rewrites (the copy, the compressor as the same derivation-conditioned operator in normal and reasoning mode, and the
earlier two-input / fact-reading proposals) are scored by the compressor reading the PRIVILEGED corrected inputs of
the whole unit, under anchored semantics (only the corrected claim changed).  The target is the arg-max, ties broken
by the smallest edit.  Every changed pair yields an exact retraction pair.  No human labels.

  python commutator21.py relabel --out models/d21                  # verified (delta, D, s) -> (D', s') pairs
  python commutator21.py data --src models/d21                     # tnet21, tnet21nd, tnet21w, tnet21nv, tnet21nr
  python commutator21.py train --out models/tnet21 --max_len 2048  # LoRA distillation (commutator.cmd_train)
  python commutator21.py eval --out models/tnet21 --served tnet21 --port 18432 --suffix _v21
"""
import argparse
import asyncio
import collections
import json
import math
import os
import re

import commutator
from commutator import T_PROMPT, W_PROMPT, render_wfacts, first_line, out_budget, norm, edit_ratio, REWRITE, LEDGER

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")

D_PROMPT = """A memory sentence was derived from facts, as its derivation record shows: each fact's id with the value the sentence took from it, and the computation for any total, count, maximum or minimum, difference, order, duration or conclusion. One of these facts has now been corrected.
Correction: {old} => {new}
Derivation record: {rec}
Sentence: {sent}
Rewrite the derivation record and the sentence as they should read after the correction: put the corrected value in place of the old one, redo the computation with the other values unchanged, and state the result in the sentence. If the sentence does not depend on the corrected fact, copy both unchanged.
Output exactly two lines:
Derivation: <record>
Sentence: <sentence, keeping its citations>"""

REC_UPDATE = """A memory sentence was rewritten after one of the facts it was derived from was corrected. Update its derivation record (each fact's id with the value taken from it, and the computation).
Correction: {old} => {new}
Derivation record before: {rec}
Sentence before: {sent}
Sentence after: {new_sent}
Write the derivation record of the sentence after the correction: the same facts, the corrected value in place of the old one, and the computation redone so that it yields what the new sentence states. Output only the record, on one line."""

VERIFY2 = """Facts of a memory unit, after one of them was corrected. Only the corrected fact changed; every other fact is exactly as listed.
{facts}

Correction: {old} => {new}
Memory sentence written before the correction: {sent}
Candidate for the sentence after the correction: {cand}

Is the candidate exactly what the sentence should say after the correction? Answer yes only if everything it states agrees with the listed facts (including totals, counts, extremes, orders, durations and conclusions that follow from them), no value replaced by the correction remains, nothing the sentence said is lost, and nothing is changed that the listed facts do not change. Answer yes or no."""

_ID = re.compile(r"\bc\d+\b")


def record_ids(rec):
    return set(_ID.findall(rec or ""))


def anchor_record(rec, oid, nid):
    """Anchoring of the record: the successor claim takes the identifier of the claim it revises."""
    return re.sub(rf"\b{oid}\b", nid, rec) if rec else rec


_OPERAND = re.compile(r"^\s*\[?(c\d+)\]?\s*[=:]")


def record_entries(rec):
    """A record is a list of entries separated by ';': operand entries "cN = <what the sentence took from claim cN>"
    and computation entries, which carry no key.  Returns [(claim id or None, entry)]."""
    out = []
    for p in (rec or "").split(";"):
        p = p.strip()
        if p:
            m = _OPERAND.match(p)
            out.append((m.group(1) if m else None, p))
    return out


def is_keyed(rec):
    """Every claim the record names has its own operand entry."""
    keys = {k for k, _ in record_entries(rec) if k}
    return bool(keys) and record_ids(rec) <= keys


def upsert_record(rec, out, cid, comp=True):
    """Keyed maintenance of a record.  The record is an auxiliary view keyed by claim: one entry per operand, and
    the computation over the operands.  A correction of claim `cid` is an upsert of that key.  The entries of `cid`
    and the computation are taken from the operator's output `out`; the entries of every other claim are carried
    over from the stored record `rec` bit for bit, because the correction did not touch those claims (anchoring).
    A maintenance error can therefore enter only the entry of the corrected claim, where the next correction of that
    claim overwrites it, and never the entries of the other operands: the error of a record after any number of
    corrections is bounded by its fan-in, not by the length of the sequence (Theorem 3).
    comp=False keeps the computation of `rec` as well and writes the entry of `cid` alone."""
    if not rec or rec.strip().lower() == "none" or not out or out.strip().lower() == "none":
        return out
    if norm(out) == norm(rec):
        return rec
    if not is_keyed(rec):
        return out                                   # a record without keys has no entry to address: it is replaced
    old, new = record_entries(rec), record_entries(out)
    mine = [p for k, p in new if k == cid]
    res, placed = [], False
    for k, p in old:
        if k is None:
            continue
        if k != cid:
            res.append(p)
        elif not placed:
            res += mine
            placed = True
    if not placed:
        res += mine                                  # the corrected claim becomes an operand of the sentence
    res += [p for k, p in (new if comp else old) if k is None]     # the computation, redone by the operator
    return "; ".join(res) if res else "none"


def entry_target(rec, ids, proposal, proposal_sent, sent, cid):
    """Record target of a pair whose sentence stays as it is.  A correction of an operand is an upsert of its key
    whether or not the statement survives the new value: "the user is a minor" is still true when the age goes from
    17 to 16, and the record must say 16 from then on, because the next correction of another operand reads it.
    The entry of the corrected claim is the one the compressor wrote in its own derivation-conditioned rewrite
    (`proposal`); its computation is taken with it only when that rewrite also left the sentence as it is."""
    if cid not in set(ids or []) or not proposal or not is_keyed(rec) or not any(k == cid for k, _ in record_entries(proposal)):
        return rec
    return upsert_record(rec, proposal, cid, comp=norm(proposal_sent or "") == norm(sent))


def foreign_entries(rec, cid):
    return [p for k, p in record_entries(rec) if k and k != cid]


def parse_two(text, rec_default, sent_default):
    """'Derivation: ...\\nSentence: ...' -> (record, sentence); a bare line is a sentence."""
    rec, sent = None, None
    for line in (text or "").splitlines():
        l = line.replace("**", "").strip()
        m = re.match(r"(?i)^(derivation( record)?|record)\s*:\s*(.*)$", l)
        if m and rec is None:
            rec = m.group(3).strip().strip('"')
            continue
        m = re.match(r"(?i)^sentence\s*:\s*(.*)$", l)
        if m and sent is None:
            sent = m.group(1).strip().strip('"')
            continue
        if rec is None and sent is None and l and not l.lower().startswith(("derivation", "sentence")):
            sent = l.strip('"')
    if sent is not None and sent == "":
        sent = None
    return (rec if rec else rec_default), (sent if sent else sent_default)


def out_budget_d(sent, rec):
    return min(int((len(sent) + len(rec or "")) / 3.2) + 64, 1200)


def two_lines(rec, sent):
    return f"Derivation: {rec}\nSentence: {sent}"


def dprompt(e):
    return D_PROMPT.format(old=e["old"], new=e["new"], rec=e.get("D") or "none", sent=e["sent"])


async def _ret(x):
    return x


# ------------------------------------------------------------------------------------------------------- relabel
async def _relabel(a):
    from cocomem import CoCoMem, strip_cites, cited_ids
    from commutator import unit_facts
    from deriv import attach as attach_deriv
    from llm import LLM, Usage, gather_limited, ContextOverflow, _overflow
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=a.conc)
    sem = asyncio.Semaphore(160)
    os.makedirs(a.out, exist_ok=True)
    cost = {"gen": Usage(), "verify": Usage()}
    si, sn = (int(x) for x in a.shard.split(":")) if a.shard else (0, 1)
    sfx = f".s{si}of{sn}" if sn > 1 else ""
    cost_path = f"{a.out}/label_cost{sfx}.json"
    if os.path.exists(cost_path):
        for key, previous in json.load(open(cost_path)).items():
            if key in cost:
                cost[key].calls, cost[key].prompt_tokens, cost[key].completion_tokens = previous["calls"], previous["in"], previous["out"]

    def save_cost():
        json.dump({k: v.as_dict() for k, v in cost.items()}, open(cost_path, "w"))

    async def p_yes(prompt):
        async with sem:
            client = llm.next_client()
            async with llm.reserved(client, prompt, 1):
                for attempt in range(6):
                    try:
                        r = await client.chat.completions.create(
                            model="qwen3-8b", messages=[{"role": "user", "content": prompt}], max_tokens=1,
                            temperature=0.0, logprobs=True, top_logprobs=10,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                        break
                    except Exception as exc:
                        if _overflow(exc):
                            raise ContextOverflow(str(exc)[:300]) from None
                        if attempt == 5:
                            raise
                        await asyncio.sleep(5 * (attempt + 1))
        cost["verify"].add(r.usage.prompt_tokens, r.usage.completion_tokens)
        py = pn = 0.0
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            w = t.token.strip().lower()
            if w == "yes":
                py += math.exp(t.logprob)
            elif w == "no":
                pn += math.exp(t.logprob)
        return py / (py + pn) if py + pn > 0 else 0.5

    mems = {}

    def mem_of(qid):
        if qid not in mems:
            m = CoCoMem.from_json(llm, json.load(open(f"{ROOT}/states/cocomem/{qid}.json")))
            attach_deriv(m, f"{ROOT}/states/deriv/{qid}.json")
            mems[qid] = m
        return mems[qid]

    def copy_placeholder(e):
        """A pair whose teacher calls did not fit the server's context window: kept as a copy with the flag
        `deferred`, to be re-labelled by `relabel --redo-deferred` on a server with the full window."""
        e["deferred"] = 1
        e["td"], e["td_rec"], e["tr"], e["tr_rec"] = e["sent"], e["D"], e["sent"], e["D"]
        e["target"], e["target_rec"], e["y"], e["scores"] = e["sent"], e["D"], 0, {}
        e["target_nv"], e["target_nv_rec"], e["y_nv"] = e["sent"], e["D"], 0
        return e

    async def cand(e):
        cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
        prev = {norm(e[k]) for k in ("old_target", "old_target5", "old_target7", "old_t_ledger", "old_tw", "old_tt")
                if e.get(k) and e[k] != LEDGER}
        e["active"] = int(cid in e["Dids"] or cid in cited_ids(e["sent"]) or len(prev - {norm(e["sent"])}) > 0)
        if not e["active"]:
            e["td"], e["td_rec"], e["tr"], e["tr_rec"] = e["sent"], e["D"], e["sent"], e["D"]
            return e
        p = dprompt(e)
        try:
            if a.think_budget > 0:
                td, tr = await asyncio.gather(
                    llm.chat(p, max_tokens=out_budget_d(e["sent"], e["D"]), usage=cost["gen"]),
                    llm.chat(p, max_tokens=out_budget_d(e["sent"], e["D"]) + a.think_budget, think=True, usage=cost["gen"]))
            else:                            # record-conditioned proposal of the compressor in normal mode only
                td = await llm.chat(p, max_tokens=out_budget_d(e["sent"], e["D"]), usage=cost["gen"])
                tr = td
            if a.propose_more and not any(e.get(k) for k in ("old_target", "old_tw", "old_tt")):
                # the other cheap paths around the square (as in v21, where they were carried over from v20): the
                # compressor as the operator that reads the record's claims dereferenced in the ledger, and as the
                # two-input operator; both are candidates for the verifier, neither is required to win
                m = mem_of(e["qid"])
                ids = sorted(record_ids(e["D"]) | set(cited_ids(e["sent"])), key=lambda c: int(c[1:]))
                lines = m._witness_lines(ids, exclude={cid})
                calls = [llm.chat(T_PROMPT.format(old=e["old"], new=e["new"], sent=e["sent"]), max_tokens=out_budget(e["sent"]), usage=cost["gen"])]
                if lines:
                    calls.append(llm.chat(W_PROMPT.format(old=e["old"], new=e["new"], facts=render_wfacts(lines), sent=e["sent"]),
                                          max_tokens=out_budget(e["sent"]), usage=cost["gen"]))
                outs = await asyncio.gather(*calls)
                e["old_tt"] = first_line(outs[0])
                if lines:
                    e["old_tw"] = first_line(outs[1])
        except ContextOverflow:
            return copy_placeholder(e)
        e["td_rec"], e["td"] = parse_two(td, e["D"], e["sent"])
        e["tr_rec"], e["tr"] = parse_two(tr, e["D"], e["sent"])
        return e

    async def verify(e):
        if e.get("deferred"):
            return e
        s = e["sent"]
        try:
            for key in ("td", "tr"):                 # sentence-only contract: a proposal must be a statement, not a comment
                proposal = e.get(key) or s
                if norm(proposal) != norm(s):
                    verdict = await llm.chat(REWRITE.format(a=strip_cites(s), b=strip_cites(proposal)), max_tokens=4, usage=cost["verify"])
                    if "yes" not in verdict.lower():
                        e[key + "_rejected"] = proposal
                        e[key], e[key + "_rec"] = s, e["D"]
            txt = {}
            for c in (s, e.get("td"), e.get("tr"), e.get("old_target"), e.get("old_target5"), e.get("old_target7"),
                      e.get("old_t_ledger"), e.get("old_tw"), e.get("old_tt")):
                if c and c != LEDGER and norm(c) and norm(c) not in txt:
                    txt[norm(c)] = c
            nv = e.get("tr") if norm(e.get("tr") or s) != norm(s) else e.get("td")
            e["target_nv"], e["y_nv"] = nv, int(norm(nv) != norm(s))
            e["target_nv_rec"] = e["tr_rec"] if norm(e.get("tr") or s) != norm(s) else e["td_rec"]
            if len(txt) <= 1:
                e["target"], e["target_rec"], e["y"], e["scores"] = s, e["D"], 0, {}
                return e
            m = mem_of(e["qid"])
            u = m.units.get(e["uid"])
            cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
            facts = unit_facts(m, u, cid, e["new"])
            old = re.sub(r"^\[c\d+\]\s*", "", e["old"])
            keys = list(txt)
            sc = dict(zip(keys, await asyncio.gather(*(p_yes(VERIFY2.format(facts=facts, old=old, new=e["new"], sent=strip_cites(s),
                                                                               cand=strip_cites(txt[k_]))) for k_ in keys))))
            best = max(keys, key=lambda k_: (sc[k_], -edit_ratio(s, txt[k_])))
            e["target"], e["y"], e["scores"] = txt[best], int(best != norm(s)), sc
            src = [n for n, c in (("copy", s), ("td", e.get("td")), ("tr", e.get("tr")), ("v20", e.get("old_target")),
                                  ("tnet5", e.get("old_target5")), ("tnet7", e.get("old_target7")), ("ledger", e.get("old_t_ledger")),
                                  ("tw", e.get("old_tw")), ("tt", e.get("old_tt"))) if c and norm(c) == best][:1]
            e["target_src"] = src
            if not e["y"]:
                e["target_rec"] = e["D"]
            elif src and src[0] == "td":
                e["target_rec"] = e["td_rec"]
            elif src and src[0] == "tr":
                e["target_rec"] = e["tr_rec"]
            else:                                    # winner from a proposer without a record: the compressor updates it
                r = await llm.chat(REC_UPDATE.format(old=e["old"], new=e["new"], rec=e["D"], sent=s, new_sent=e["target"]),
                                   max_tokens=out_budget_d("", e["D"]) + 40, usage=cost["gen"])
                e["target_rec"] = first_line(r) or e["D"]
            return e
        except ContextOverflow:
            return copy_placeholder(e)

    for name in ("train", "test"):
        final = f"{a.out}/{name}{sfx}.jsonl"
        if os.path.exists(f"{a.out}/{name}.jsonl") or os.path.exists(final):
            if not a.redo_deferred:
                print(name, "already verified", flush=True)
                continue
            # second pass on a server with the full context window: only the deferred pairs are re-labelled
            path = final if os.path.exists(final) else f"{a.out}/{name}.jsonl"
            E_all = [json.loads(l) for l in open(path)]
            redo = [i for i, e in enumerate(E_all) if e.get("deferred")]
            print(name, "deferred pairs to redo", len(redo), flush=True)
            if not redo:
                continue
            E, unfixed = [], 0
            for i in redo:
                e = dict(E_all[i])
                for key in ("deferred", "active", "td", "td_rec", "tr", "tr_rec", "td_rejected", "tr_rejected", "target",
                            "target_rec", "y", "scores", "target_nv", "target_nv_rec", "y_nv", "target_src"):
                    e.pop(key, None)
                if e.get("no_record"):          # the unit's record has been written since: take it up
                    u = mem_of(e["qid"]).units.get(e["uid"])
                    if u is not None and "deriv" in u and e["k"] < len(u["sents"]) and norm(u["sents"][e["k"]]) == norm(e["sent"]):
                        e["D"], e["Dids"] = u["deriv"][e["k"]], list(u["dids"][e["k"]])
                        e.pop("no_record")
                    else:
                        unfixed += 1
                E.append(e)
            if unfixed:
                print(name, "pairs still without a record:", unfixed, flush=True)
            E = await gather_limited([cand(e) if not e.get("no_record") else _ret(copy_placeholder(e)) for e in E],
                                     desc=f"dcand-redo-{name}", every=500)
            R = await gather_limited([verify(e) for e in E], desc=f"dverify-redo-{name}", every=500)
            for i, e in zip(redo, R):
                E_all[i] = e
            with open(path + ".tmp", "w") as f:
                for e in E_all:
                    f.write(json.dumps(e) + "\n")
            os.replace(path + ".tmp", path)
            save_cost()
            print(name, "redo done; still deferred", sum(bool(e.get("deferred")) for e in E_all), flush=True)
            continue
        ck = f"{a.out}/{name}.cand{sfx}.jsonl"
        if os.path.exists(ck):
            E = [json.loads(l) for l in open(ck)]
        else:
            # pairs of v21 = (correction, sentence) over the v21 sentence segmentation; earlier proposals are carried
            # over to the sentences that are unchanged by the segmentation (matched by normalized text)
            old = [json.loads(l) for l in open(f"{a.src}/{name}.jsonl")]
            old = [e for e in old if not e.get("rt")]
            groups = {}
            for e in old:
                groups.setdefault((e["qid"], e["uid"], e["old"], e["new"]), []).append(e)
            E, stats = [], {"pairs": 0, "carried": 0, "merged": 0, "no_unit": 0}
            for (qid, uid, oldc, newc), es in groups.items():
                m = mem_of(qid)
                u = m.units.get(uid)
                if u is None:
                    stats["no_unit"] += 1
                    continue
                c0 = re.match(r"\[(c\d+)\]", oldc)
                if c0 and c0.group(1) not in u["inputs"]:
                    # the unit was not written from this claim (a record-first profile is written from the claims
                    # its selection call kept): the unit is outside the closure of the correction
                    stats["outside"] = stats.get("outside", 0) + 1
                    continue
                by = {norm(e["sent"]): e for e in es}
                if "deriv" in u:
                    rows = list(zip(u["sents"], u["deriv"], u["dids"]))
                    no_record = False
                else:                            # record not written yet (profile units wait for a full-window
                    from deriv import unit_sents   # server): the pair is deferred and completed by --redo-deferred
                    rows = [(s, "none", []) for s in unit_sents(u)]
                    no_record = True
                    stats["no_record"] = stats.get("no_record", 0) + 1
                for k, (s, d, ids) in enumerate(rows):
                    e0 = by.get(norm(s))
                    x = {"qid": qid, "uid": uid, "level": u["level"], "k": k, "old": oldc, "new": newc, "sent": s,
                         "D": d, "Dids": list(ids)}
                    if no_record:
                        x["no_record"] = 1
                    if e0 is not None:
                        stats["carried"] += 1
                        for key in ("target", "target5", "target7", "t_ledger", "tw", "tt", "W"):
                            if e0.get(key) is not None:
                                x["old_" + key] = e0[key]
                    else:
                        stats["merged"] += 1
                    E.append(x)
                    stats["pairs"] += 1
            print(name, "pairs", json.dumps(stats), flush=True)
            if a.limit:                      # smoke tests only
                E = E[:a.limit]
            if sn > 1:                       # shard i of n: every n-th pair (the shards are merged by `merge`)
                E = E[si::sn]
            E = await gather_limited([cand(e) if not e.get("no_record") else _ret(copy_placeholder(e)) for e in E],
                                     desc=f"dcand-{name}", every=2000)
            with open(ck + ".tmp", "w") as f:
                for e in E:
                    f.write(json.dumps(e) + "\n")
            os.replace(ck + ".tmp", ck)
            save_cost()

        R = await gather_limited([verify(e) for e in E], desc=f"dverify-{name}", every=2000)
        with open(f"{a.out}/{name}{sfx}.jsonl.tmp", "w") as f:
            for e in R:
                f.write(json.dumps(e) + "\n")
        os.replace(f"{a.out}/{name}{sfx}.jsonl.tmp", f"{a.out}/{name}{sfx}.jsonl")
        src = {}
        for e in R:
            for x in e.get("target_src") or (["copy"] if not e["y"] else []):
                src[x] = src.get(x, 0) + 1
        rec_hit = sum(re.match(r"\[(c\d+)\]", e["old"]).group(1) in set(e["Dids"]) for e in R if e["y"])
        print(name, len(R), "pos", sum(e["y"] for e in R), "pos_nv", sum(e["y_nv"] for e in R), "active", sum(e["active"] for e in R),
              "deferred", sum(bool(e.get("deferred")) for e in R),
              "record_recall", round(rec_hit / max(1, sum(e["y"] for e in R)), 3), "src", json.dumps(src), flush=True)
        save_cost()


def cmd_merge(a):
    """Merge the shards written by `relabel --shard i:n` into {train,test}.jsonl and one label_cost.json."""
    n = a.shards
    for name in ("train", "test"):
        if os.path.exists(f"{a.out}/{name}.jsonl"):
            continue
        parts = [f"{a.out}/{name}.s{i}of{n}.jsonl" for i in range(n)]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"missing shards: {missing}")
        rows = [[json.loads(l) for l in open(p)] for p in parts]
        deferred = sum(bool(e.get("deferred")) for r in rows for e in r)
        if deferred:
            raise SystemExit(f"{name}: {deferred} deferred pairs remain; run relabel --redo-deferred on a full-window server first")
        out = []
        for k in range(max(map(len, rows))):           # interleave back into the original order
            for r in rows:
                if k < len(r):
                    out.append(r[k])
        with open(f"{a.out}/{name}.jsonl.tmp", "w") as f:
            for e in out:
                f.write(json.dumps(e) + "\n")
        os.replace(f"{a.out}/{name}.jsonl.tmp", f"{a.out}/{name}.jsonl")
        print(name, len(out), "pos", sum(e["y"] for e in out), flush=True)
    cost = {"gen": {"calls": 0, "in": 0, "out": 0}, "verify": {"calls": 0, "in": 0, "out": 0}}
    for i in range(n):
        p = f"{a.out}/label_cost.s{i}of{n}.json"
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                for kk in cost[k]:
                    cost[k][kk] += v.get(kk, 0)
    json.dump(cost, open(f"{a.out}/label_cost.json", "w"))


# ------------------------------------------------------------------------------------------------------- data
def cmd_data(a):
    """Training sets of the v21 student and its ablations (a.src/{train,test}.jsonl -> models/<variant>/).

      tnet21    (delta, D, s) -> (D', s')   verified targets, + retraction pairs           (CoCoMem)
      tnet21nd  (delta, s)    -> s'         same targets and pairs, two-input student       (- record)
      tnet21w   (delta, s, deref(ids(D)))  -> s'   pointer witness dereferenced in the ledger (v20 architecture)
      tnet21nv  (delta, D, s) -> (D', s')   UNVERIFIED reasoning proposals as targets        (- verification)
      tnet21nr  (delta, D, s) -> (D', s')   verified targets, NO retraction pairs            (- retraction)

    Retraction pairs: if (D', s') repairs (D, s) for delta = (c -> c'), then the commutative repair of (D', s') for
    delta^-1 = (c' -> c) is exactly (D, s), the compressor's own output for H.  No teacher call, no label noise."""
    from cocomem import CoCoMem
    tr = [json.loads(l) for l in open(f"{a.src}/train.jsonl")]
    te = [json.loads(l) for l in open(f"{a.src}/test.jsonl")]
    mems = {}

    def deref(qid, ids, cid):
        if qid not in mems:
            mems[qid] = CoCoMem.from_json(None, json.load(open(f"{ROOT}/states/cocomem/{qid}.json")))
        m = mems[qid]
        return [f"[{c}] ({m.claims[c]['date'][:10]}, {m.claims[c]['speaker']}) {m.claims[c]['text']}"
                for c in ids if c != cid and c in m.claims]

    P_ = a.prefix                                     # tnet21 (v21 memories) or tnet22 (record-first memories)
    variants = {P_: ("rec", "target", True), P_ + "nd": ("none", "target", True), P_ + "w": ("ptr", "target", True),
                P_ + "nv": ("rec", "target_nv", True), P_ + "nr": ("rec", "target", False)}

    def ex(e, mode, tkey):
        t = e[tkey]
        trec = e["target_rec"] if tkey == "target" else e["target_nv_rec"]
        x = {k: e[k] for k in ("qid", "uid", "level", "k", "old", "new", "sent", "D", "Dids") if k in e}
        x["target_sent"], x["y"] = t, int(norm(t) != norm(e["sent"]))
        if mode == "rec":
            x["target_rec"] = trec if x["y"] else e["D"]
            if a.keyed and x["y"]:
                # the record target as it is stored: the entries of the uncorrected operands are those of the input
                # record (upsert_record), so the student is trained on the keyed write it is deployed with
                cid = re.match(r"\[(c\d+)\]", x["old"]).group(1)
                kept = upsert_record(e["D"], x["target_rec"], cid)
                stats["rec targets"] += 1
                stats["rec targets with an altered entry of another claim"] += foreign_entries(x["target_rec"], cid) != foreign_entries(e["D"], cid)
                x["target_rec"] = kept
            elif a.keyed:
                # the sentence stays; if the corrected claim is one of its operands, its entry is still written
                cid = re.match(r"\[(c\d+)\]", x["old"]).group(1)
                stats["copies whose record names the corrected claim"] += cid in set(e.get("Dids") or [])
                ent = entry_target(e["D"], e.get("Dids"), e.get("td_rec"), e.get("td"), e["sent"], cid)
                if norm(ent) != norm(e["D"]):
                    x["target_rec"], x["yrec"] = ent, 1
                    stats["copies with the entry of the corrected claim rewritten"] += 1
            x["target"] = two_lines(x["target_rec"], t)
            x["prompt"] = dprompt(x)
        elif mode == "none":
            x["target"] = t
            x["prompt"] = T_PROMPT.format(old=x["old"], new=x["new"], sent=x["sent"])
        else:
            cid = re.match(r"\[(c\d+)\]", x["old"]).group(1)
            x["wfacts"] = render_wfacts(deref(x["qid"], x["Dids"], cid))
            x["target"] = t
            x["prompt"] = W_PROMPT.format(old=x["old"], new=x["new"], facts=x["wfacts"], sent=x["sent"])
        return x

    def inverse(x, mode):
        cid = re.match(r"\[(c\d+)\]", x["old"]).group(1)
        old_plain = re.sub(r"^\[c\d+\]\s*", "", x["old"])
        y = dict(x, old=f"[{cid}] {x['new']}", new=old_plain, sent=x["target_sent"], target_sent=x["sent"], rt=1, y=1)
        if mode == "rec":
            y["D"], y["target_rec"] = x["target_rec"], x["D"]
            y["target"] = two_lines(x["D"], x["sent"])
            y["prompt"] = dprompt(y)
        elif mode == "none":
            y["target"] = x["sent"]
            y["prompt"] = T_PROMPT.format(old=y["old"], new=y["new"], sent=y["sent"])
        else:
            y["target"] = x["sent"]
            y["prompt"] = W_PROMPT.format(old=y["old"], new=y["new"], facts=x["wfacts"], sent=y["sent"])
        return y

    only = set(a.variants.split(",")) if getattr(a, "variants", "") else None
    for name, (mode, tkey, rt) in variants.items():
        if only and name not in only:
            continue
        d = f"{os.path.dirname(a.src.rstrip('/'))}/{name}"
        os.makedirs(d, exist_ok=True)
        stats = collections.Counter()
        T = [ex(e, mode, tkey) for e in tr]
        if stats:
            print(name, "keyed record targets:", dict(stats), flush=True)
        if rt:
            # the inverse of a pair is exact whenever its target differs from its input, in the sentence or in the record
            T += [inverse(x, mode) for x in T if x["y"] or x.get("yrec")]
        with open(f"{d}/train.jsonl", "w") as f:
            for x in T:
                f.write(json.dumps(x) + "\n")
        with open(f"{d}/test.jsonl", "w") as f:              # one held-out set for every variant: verified targets
            for e in te:
                f.write(json.dumps(ex(e, mode, "target")) + "\n")
        print(name, "train", len(T), "pos", sum(x["y"] for x in T), "retraction", sum(bool(x.get("rt")) for x in T),
              "test", len(te), flush=True)


# ------------------------------------------------------------------------------------------------------- eval
async def _eval(a):
    from cfgen import SAME
    from cocomem import strip_cites
    from llm import LLM, gather_limited
    E = [json.loads(l) for l in open(f"{a.out}/test.jsonl")]
    T = LLM(a.served, a.port, concurrency=192)
    J = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=128)
    rec_mode = all("Sentence:" in e["target"] for e in E[:50])

    async def one(e):
        raw = await T.chat(e["prompt"], max_tokens=out_budget_d(e["sent"], e.get("D")) if rec_mode else out_budget(e["sent"]))
        if rec_mode:
            rec, out = parse_two(raw, e.get("D"), e["sent"])
        else:
            rec, out = None, first_line(raw)
        changed = norm(out) != norm(e["sent"])
        good = None
        if e["y"] and changed:
            good = norm(out) == norm(e["target_sent"]) or "no" not in (await J.chat(
                SAME.format(a=strip_cites(e["target_sent"]), b=strip_cites(out)), max_tokens=4)).lower()
        rec_ok = None
        if rec_mode and e["y"]:
            rec_ok = norm(rec or "") == norm(e.get("target_rec") or "")
        return {"changed": changed, "good": good, "out": out, "rec": rec, "rec_ok": rec_ok}

    R = await gather_limited([one(e) for e in E], desc="eval", every=1000)
    tp = sum(r["changed"] and e["y"] for r, e in zip(R, E))
    fp = sum(r["changed"] and not e["y"] for r, e in zip(R, E))
    fn = sum((not r["changed"]) and e["y"] for r, e in zip(R, E))
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = {"p": p, "r": rc, "f1": 2 * p * rc / max(1e-9, p + rc), "rewrite_ok": sum(bool(r["good"]) for r in R) / max(1, tp),
           "exact_commute": sum((bool(r["good"]) if e["y"] else not r["changed"]) for r, e in zip(R, E)) / len(E)}
    if rec_mode:
        pos = [r for r, e in zip(R, E) if e["y"]]
        res["rec_exact"] = sum(bool(r["rec_ok"]) for r in pos) / max(1, len(pos))
    for lev in ("note", "digest", "profile"):
        ii = [i for i, e in enumerate(E) if e["level"] == lev]
        tp_ = sum(R[i]["changed"] and E[i]["y"] for i in ii); fp_ = sum(R[i]["changed"] and not E[i]["y"] for i in ii)
        fn_ = sum((not R[i]["changed"]) and E[i]["y"] for i in ii)
        res[lev] = {"p": tp_ / max(1, tp_ + fp_), "r": tp_ / max(1, tp_ + fn_)}
    print(json.dumps(res), flush=True)
    json.dump({"res": res, "out": R}, open(f"{ROOT}/results/tnet_eval_{a.served}{a.suffix}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--src", default=f"{ROOT}/models/w20", help="relabel: v20 pairs (proposals carried over); data: v21 pairs")
    ap.add_argument("--out", default=f"{ROOT}/models/d21")
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--think_budget", type=int, default=0, help="relabel: >0 adds a reasoning-mode proposal with this token budget")
    ap.add_argument("--limit", type=int, default=0, help="relabel: at most N pairs per split (smoke tests)")
    ap.add_argument("--shard", default="", help="relabel: i:n -- label every n-th pair starting at i (parallel GPUs)")
    ap.add_argument("--redo-deferred", action="store_true", help="relabel: re-label the pairs deferred by a reduced-context server")
    ap.add_argument("--propose-more", action="store_true",
                    help="relabel: also propose with the compressor as the claim-reading and the two-input operator (v22 memories carry no v20 proposals)")
    ap.add_argument("--shards", type=int, default=1, help="merge: number of shards")
    ap.add_argument("--prefix", default="tnet21", help="data: name of the main student; ablations get the suffixes nd/w/nv/nr")
    ap.add_argument("--keyed", action="store_true", help="data: record targets keep the entries of the uncorrected operands of the input record")
    ap.add_argument("--variants", default="", help="data: write only these training sets (comma-separated names)")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=commutator.BASE)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--tok_budget", type=int, default=4096)
    ap.add_argument("--qlora", action="store_true", help="train: 4-bit NF4 frozen base (same LoRA recipe), merged into the bf16 base")
    ap.add_argument("--merge-only", action="store_true", help="train: merge the adapter saved by an interrupted run (adapter/train_stats.json)")
    ap.add_argument("--served", default="tnet21")
    ap.add_argument("--port", type=int, default=18432)
    ap.add_argument("--suffix", default="_v21")
    a = ap.parse_args()
    if a.cmd == "relabel":
        asyncio.run(_relabel(a))
    elif a.cmd == "merge":
        cmd_merge(a)
    elif a.cmd == "data":
        cmd_data(a)
    elif a.cmd == "train":
        commutator.BASE, commutator.OUT = a.base, a.out
        commutator.PROMPT = commutator.prompt_of
        commutator.cmd_train(a)
    elif a.cmd == "eval":
        asyncio.run(_eval(a))
