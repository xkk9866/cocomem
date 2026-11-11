"""Materialized derivation records (v21): the auxiliary view that makes every stored sentence self-maintainable
WITHOUT base access.

Lemma 1 (indistinguishability): a sentence that merged several facts ("owns 5 plants" from 3+2 or from 4+1) cannot be
repaired from itself and the correction.  Incremental view maintenance answers this with an auxiliary view that stores
the base data the view needs (Quass et al. 1996; Gupta, Jagadish & Mumick 1996): the view then becomes
self-maintainable in the strict sense, i.e. maintainable from the view, the auxiliary view and the delta alone.
v20 stored a *pointer* witness (claim identifiers) and dereferenced it in the ledger at correction time; that is a
join with the base table, it costs the full claim texts per call, and its recall depended on a post-hoc attribution.
v21 materializes the derivation itself: for every sentence k the compressor records, once, at write time,
    D_k = the claims the sentence was derived from, each with the VALUE the sentence took from it,
          and the computation (total, count, extreme, difference, order, duration, conclusion).
The record is a few dozen tokens, is stored next to the unchanged unit text, and is anchored like the unit's inputs.
At correction time the operator reads only (delta, D_k, s_k): no claim is read from the ledger.

  python deriv.py annotate --kind lme --required-only   # states/deriv/<qid>.json for the v21 training/evaluation users
  python deriv.py annotate --kind locomo                # states/deriv_locomo/<cid>.json
  python deriv.py stats --pairs models/tnet21/test.jsonl --out deriv_stats.json
"""
import argparse
import asyncio
import glob
import json
import os
import re

ROOT = os.environ.get("CM_RUN_ROOT", "/data/yaokang/corrmem")
DDIR = {"lme": f"{ROOT}/states/deriv", "locomo": f"{ROOT}/states/deriv_locomo"}
SDIR = {"lme": f"{ROOT}/states/cocomem", "locomo": f"{ROOT}/states/locomo_cocomem"}

DERIV = """You maintain a memory {kind} of a personal assistant. It was written from the facts below.

Facts:
{facts}

The {kind}, sentence by sentence:
{unit}

For every sentence, write its derivation record: the facts it was derived from, each as the fact id, "=", and the exact value or detail the sentence takes from that fact (a number, name, date, place, item or short statement), separated by semicolons. If the sentence states a total, a count, a maximum or minimum, a difference, a first or last event, an order, a duration or a conclusion, list EVERY fact that enters it, including facts the sentence does not name, and end the record with the computation and its result, for example "total 3 + 2 = 5", "max = 52 km (Monday)", "count above 20 = 3 of 5", "2023/05 to 2024/02 = 9 months", "17 < 18, so a minor". Write "none" for a sentence that uses none of the facts.

Output exactly one line per sentence, in the form "<k>: <record>", and nothing else."""

# v2 of the record prompt: the same instructions with the output grammar made explicit by example (one fact per
# entry, "id = value", every operand of an aggregate, then the computation).  Selected with CM_DERIV_PROMPT=v2.
DERIV_V2 = """You maintain a memory {kind} of a personal assistant. It was written from the facts below.

Facts:
{facts}

The {kind}, sentence by sentence:
{unit}

For every sentence, write its derivation record: the facts it was derived from, each as one entry "<fact id> = <the exact value or detail the sentence takes from that fact>" (a number, name, date, place, item or short statement), entries separated by semicolons, one fact id per entry. If the sentence states a total, a count, a maximum or minimum, a second-highest, a difference, a first or last event, an order, a duration or a conclusion, list EVERY fact that was compared or added as its own entry, including the facts that did not win the comparison and facts the sentence does not name, and end the record with the computation and its result. Write "none" for a sentence that uses none of the facts.

Output exactly one line per sentence, in the form "<k>: <record>", and nothing else. Examples of the form:
0: c3 = 3 plants; c7 = 2 plants; total 3 + 2 = 5
1: c12 = Monday 52 km; c13 = Tuesday 30 km; c15 = Friday 41 km; max = 52 km (Monday)
2: c12 = Monday 52 km; c13 = Tuesday 30 km; c15 = Friday 41 km; min = 30 km (Tuesday)
3: c4 = age 17; 17 < 18, so a minor
4: c21 = started 2023/05; c25 = ended 2024/02; 2023/05 to 2024/02 = 9 months
5: c9 = lives in Lyon
6: none"""

if os.environ.get("CM_DERIV_PROMPT", "v1") == "v2":
    DERIV = DERIV_V2

_ID = re.compile(r"\bc\d+\b")


def record_ids(rec):
    return set(_ID.findall(rec or ""))


def unit_sents(u):
    from cocomem import split_sentences_v2
    t = u["text_cited"]
    return [t[a:b].strip() for a, b in split_sentences_v2(t)]


def _parse(r, n):
    out = ["none"] * n
    for line in r.splitlines():
        m = re.match(r"\s*\(?(\d+)\)?\s*[:.)]\s*(.*)$", line)
        if not m or int(m.group(1)) >= n:
            continue
        rec = m.group(2).strip().strip('"')
        out[int(m.group(1))] = rec if rec else "none"
    return out


def _restrict(rec, allowed):
    """Keep the record text; drop entries whose id is not an input of the unit (hallucinated ids)."""
    if rec.lower() == "none":
        return "none"
    parts = [p.strip() for p in rec.split(";") if p.strip()]
    kept = [p for p in parts if not _ID.search(p) or any(c in allowed for c in _ID.findall(p))]
    return "; ".join(kept) if kept else "none"


async def annotate_unit(llm, m, u, usage=None):
    """Derivation records of every sentence of unit u, written by the compressor while it still sees all of the
    unit's inputs.  Returns (sentences, records, ids): ids are the record's claims united with the sentence's inline
    citations, restricted to the unit's anchored inputs."""
    from cocomem import cited_ids
    from llm import ContextOverflow
    sents = unit_sents(u)
    ins = [c for c in u["inputs"] if c in m.claims]
    if not sents:
        return sents, [], []
    allowed = set(ins)
    shown = "\n".join(f"({k}) {s}" for k, s in enumerate(sents))
    kind = {"note": "note", "digest": "digest", "profile": "profile"}[u["level"]]
    prompt = DERIV.format(kind=kind, facts=m._fact_lines(ins), unit=shown)
    max_tokens = min(80 + 64 * len(sents) + 4 * len(ins), 4000)
    # servers on shared GPUs may run with a reduced context window (CM_MAXLEN); a unit whose prompt does not fit is
    # deferred to a server with the full window instead of being truncated
    maxlen = int(os.environ.get("CM_MAXLEN", "0") or 0)
    if maxlen and len(prompt) // 3 + max_tokens + 64 > maxlen:
        raise ContextOverflow(f"estimated {len(prompt) // 3 + max_tokens} tokens > {maxlen}")
    r = await llm.chat(prompt, max_tokens=max_tokens, usage=usage)
    D = [_restrict(d, allowed) for d in _parse(r, len(sents))]
    ids = [sorted((record_ids(d) | {c for c in cited_ids(s) if c in allowed}) & allowed, key=lambda c: int(c[1:]))
           for d, s in zip(D, sents)]
    return sents, D, ids


async def annotate_mem(llm, m, usage=None, uids=None, deferred=None):
    """Records of the units `uids` (all units if None).  Units whose prompt does not fit the server's context are
    left out (and counted in `deferred`, a list) so that a server with the full window annotates them later."""
    from llm import ContextOverflow
    uids = list(m.units) if uids is None else [x for x in uids if x in m.units]
    got = await asyncio.gather(*(annotate_unit(llm, m, m.units[x], usage) for x in uids), return_exceptions=True)
    out = {}
    for x, g in zip(uids, got):
        if isinstance(g, ContextOverflow):
            if deferred is not None:
                deferred.append(x)
            continue
        if isinstance(g, BaseException):
            raise g
        s, d, i = g
        out[x] = {"sents": s, "deriv": d, "ids": i, "text_cited": m.units[x]["text_cited"]}
    return out


def pair_units(paths):
    """{qid: set(uid)} of the units that occur in the given (delta, sentence) pair files."""
    want = {}
    for p in [x for x in paths.split(",") if x]:
        with open(p) as stream:
            for line in stream:
                e = json.loads(line)
                want.setdefault(e["qid"], set()).add(e["uid"])
    return want


def attach(m, path):
    """Attach precomputed records to a loaded memory (only to units whose text is unchanged since annotation)."""
    if not os.path.exists(path):
        return 0
    d = json.load(open(path))["units"]
    n = 0
    for uid, v in d.items():
        u = m.units.get(uid)
        if u is not None and u["text_cited"] == v["text_cited"] and len(v["sents"]) == len(v["deriv"]):
            u["sents"], u["deriv"], u["dids"] = list(v["sents"]), list(v["deriv"]), [list(x) for x in v["ids"]]
            n += 1
    return n


def state_files(kind, required_only=False):
    files = sorted(glob.glob(f"{SDIR[kind]}/*.json"))
    if kind == "lme" and required_only:
        from splits import add_test200, make_splits
        qids = set(add_test200(make_splits())["test200"])
        for split in ("train", "test"):
            with open(f"{ROOT}/models/tnet10/{split}.jsonl") as stream:
                qids.update(json.loads(line)["qid"] for line in stream)
        files = [f for f in files if os.path.basename(f)[:-5] in qids]
    return files


async def cmd_annotate(a):
    from cocomem import CoCoMem
    from llm import LLM, Usage, gather_limited
    llm = LLM("qwen3-8b", int(os.environ.get("CM_P8", 18431)), concurrency=a.conc)
    os.makedirs(DDIR[a.kind], exist_ok=True)
    files = state_files(a.kind, a.required_only)
    want = None
    if a.units_from:                              # only the units that occur in the training / held-out pairs; every
        want = pair_units(a.units_from)           # other unit is annotated on demand at correction time (same pass)
        files = [f for f in files if os.path.basename(f)[:-5] in want]
    if a.limit:
        files = files[:a.limit]
    if a.shard:                                   # "i:n": this process annotates files i, i+n, ... (parallel GPUs)
        i, n = (int(x) for x in a.shard.split(":"))
        files = files[i::n]
    sem = asyncio.Semaphore(a.inst)
    deferred = []

    def load_prev(out):
        return json.load(open(out)) if os.path.exists(out) else None

    def todo_of(state, prev, qid):
        have = set((prev or {}).get("units", {}))
        if a.profiles_only:                       # the profile prompts list every claim of the user: full window only
            return sorted(x for x, u in state["units"].items() if u["level"] == "profile" and x not in have)
        if want is None:
            if prev is not None and not prev.get("partial"):
                return []
            return sorted(x for x in state["units"] if x not in have)
        return sorted(want[qid] - have)

    async def one(f):
        qid = os.path.basename(f)[:-5]
        out = f"{DDIR[a.kind]}/{os.path.basename(f)}"
        state = json.load(open(f))
        if not todo_of(state, load_prev(out), qid):
            return "cached"
        async with sem:
            # several jobs (pair shards, profile pass) may add units to the same file: read the file right before
            # annotating and again right before writing, and merge, so that no other job's records are dropped
            todo = todo_of(state, load_prev(out), qid)
            if not todo:
                return "cached"
            m = CoCoMem.from_json(llm, state)
            u = Usage()
            units = await annotate_mem(llm, m, u, uids=todo, deferred=deferred)
            prev = load_prev(out)
            d = {"units": dict((prev or {}).get("units", {})), "usage": dict((prev or {}).get("usage") or {}),
                 "n_claims": len(m.claims), "n_units": len(m.units)}
            d["units"].update(units)
            for k, v in u.as_dict().items():
                d["usage"][k] = d["usage"].get(k, 0) + v
            d["partial"] = len(d["units"]) < len(m.units)
            json.dump(d, open(out + ".tmp", "w"))
            os.replace(out + ".tmp", out)
            return "ok" if len(units) == len(todo) else "deferred"

    res = await gather_limited([one(f) for f in files], desc=f"deriv-{a.kind}", every=20)
    print(a.kind, {r: res.count(r) for r in set(res)}, "deferred_units", len(deferred), flush=True)


def cmd_stats(a):
    """Record quality and size.  Recall: share of held-out verified positives (sentence k changes under the correction
    of claim c) whose record names c, compared with the compressor's inline citations and, when available, the v20
    pointer witness.  Fan-in (claims per record) and record length by level, and their correlation with the size of
    the user's ledger; write-time tokens per memory."""
    import numpy as np
    E = [json.loads(l) for l in open(a.pairs)]
    E = [e for e in E if not e.get("rt")]
    rec = {}
    for e in E:
        if not e["y"]:
            continue
        cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
        for lev in (e["level"], "all"):
            r = rec.setdefault(lev, [0, 0, 0, 0, 0])
            r[0] += 1
            r[1] += cid in set(e.get("Dids") or [])
            r[2] += cid in set(re.findall(r"c\d+", " ".join(re.findall(r"\[([^\]]*)\]", e["sent"]))))
            r[3] += cid in record_ids(e.get("D") or "")
            if "W" in e:
                r[4] += cid in set(e.get("W") or [])
    out = {"recall": {k: {"n": v[0], "record": 100 * v[1] / v[0], "record_only": 100 * v[3] / v[0],
                          "citations": 100 * v[2] / v[0], "pointer_witness": 100 * v[4] / v[0]} for k, v in rec.items()}}
    fan, length, size, wtok, wunits, wfull, nunits = {}, {}, [], 0, 0, [], []
    required = {os.path.basename(f) for f in state_files("lme", True)} if a.required_only else None
    for f in glob.glob(f"{DDIR['lme']}/*.json"):
        if required is not None and os.path.basename(f) not in required:
            continue
        d = json.load(open(f))
        per = []
        for uid, v in d["units"].items():
            lev = "profile" if uid.startswith("profile") else "digest" if uid.startswith("d") else "note"
            for ids, r in zip(v["ids"], v["deriv"]):
                fan.setdefault(lev, []).append(len(ids))
                length.setdefault(lev, []).append(len(r) // 4 + 1)
                per.append(len(ids))
        if per:
            size.append((d.get("n_claims", 0), float(np.mean(per)), max(per)))
        # write cost: tokens of the record pass per annotated unit (files may hold only the units of the training
        # pairs; the remaining units are annotated on demand by the same pass and reported separately)
        wtok += d["usage"]["in"] + d["usage"]["out"]
        wunits += len(d["units"])
        if not d.get("partial") and d.get("n_units", len(d["units"])) == len(d["units"]):
            wfull.append(d["usage"]["in"] + d["usage"]["out"])
    for f in state_files("lme", True):
        nunits.append(len(json.load(open(f))["units"]))
    S = np.array(size, float)
    q = np.quantile(S[:, 0], [0, .25, .5, .75, 1])
    out["fanin"] = {k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p90": float(np.quantile(v, .9)),
                        "max": int(max(v))} for k, v in fan.items()}
    out["record_tokens"] = {k: {"mean": float(np.mean(v)), "p90": float(np.quantile(v, .9)), "max": int(max(v))}
                            for k, v in length.items()}
    out["fanin_vs_history"] = {"r_mean": float(np.corrcoef(S[:, 0], S[:, 1])[0, 1]),
                               "by_quartile": [float(S[(S[:, 0] >= q[i]) & (S[:, 0] <= q[i + 1]), 1].mean()) for i in range(4)],
                               "edges": [float(x) for x in q]}
    per_unit = wtok / max(1, wunits)
    out["write_tokens_per_unit"] = float(per_unit)
    out["units_per_memory"] = float(np.mean(nunits)) if nunits else 0.0
    out["write_tokens_per_memory"] = float(per_unit * np.mean(nunits)) if nunits else 0.0
    out["write_tokens_per_memory_full_files"] = {"n": len(wfull), "mean": float(np.mean(wfull)) if wfull else 0.0}
    out["annotated_units"] = wunits
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--kind", default="lme")
    ap.add_argument("--conc", type=int, default=96)
    ap.add_argument("--inst", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i:n -- annotate every n-th file starting at i")
    ap.add_argument("--required-only", action="store_true", help="Only the training/evaluation users and the paper test split")
    ap.add_argument("--units-from", default="", help="comma-separated pair files: annotate only the (qid, uid) units they contain")
    ap.add_argument("--profiles-only", action="store_true", help="annotate (or count) only the profile units of the selected memories")
    ap.add_argument("--pairs", default=f"{ROOT}/models/tnet21/test.jsonl")
    ap.add_argument("--out", default="deriv_stats.json")
    a = ap.parse_args()
    if a.cmd == "annotate":
        asyncio.run(cmd_annotate(a))
    elif a.cmd == "stats":
        cmd_stats(a)
    elif a.cmd == "count":
        # prints "<memories> <memories whose selected units all have records> <selected units> <units with records>"
        files = state_files(a.kind, a.required_only)
        want = None
        if a.units_from:
            want = pair_units(a.units_from)
            files = [f for f in files if os.path.basename(f)[:-5] in want]
        if a.shard:
            i, n = (int(x) for x in a.shard.split(":"))
            files = files[i::n]
        done = n_sel = n_have = 0
        for f in files:
            qid = os.path.basename(f)[:-5]
            out = f"{DDIR[a.kind]}/{os.path.basename(f)}"
            have = set(json.load(open(out))["units"]) if os.path.exists(out) else set()
            if a.profiles_only:
                sel = {x for x, u in json.load(open(f))["units"].items() if u["level"] == "profile"}
            elif want is not None:
                sel = set(want[qid])
            else:
                sel = set(json.load(open(f))["units"])
            n_sel += len(sel)
            n_have += len(sel & have)
            done += not (sel - have)
        print(len(files), done, n_sel, n_have)
