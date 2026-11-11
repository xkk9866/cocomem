"""Derivation witnesses: the auxiliary view that makes every stored sentence self-maintainable.

Lemma 1 (indistinguishability) says that a sentence-local repair cannot be commutative when the sentence merged
information that the correction separates ("owns 5 plants" from 3+2 or from 4+1).  Incremental view maintenance
has a classical answer for views that are not self-maintainable: materialize an *auxiliary view* that makes them so
(Quass et al. 1996; Gupta, Jagadish & Mumick 1996).  For an LLM-written unit the auxiliary view is, per sentence k,
its *derivation witness* W_k, the set of ledger claims from which the sentence was derived (stated, counted, added,
compared, ranked, ordered in time or concluded from).  The witness is computed ONCE at write time by the compressor,
which then still sees all of the unit's inputs, and stored as claim identifiers.  At correction time the witness is
DEREFERENCED against the current ledger, so it always carries the current operands, also after earlier corrections.

  python witness.py annotate --kind lme      # states/witness/<qid>.json     for every states/cocomem/<qid>.json
  python witness.py annotate --kind locomo   # states/witness_locomo/<cid>.json
  python witness.py stats --pairs models/tnet20/test.jsonl   # recall of the witness on held-out verified positives,
                                                              # fan-in by level and against history size

The unit texts are NOT changed: the witness is metadata next to the unit, so every other maintenance strategy and
every reading of the memory is unaffected (Orig is identical for all strategies)."""
import argparse
import asyncio
import glob
import json
import os
import re

ROOT = "/data/yaokang/corrmem"
WDIR = {"lme": f"{ROOT}/states/witness", "locomo": f"{ROOT}/states/witness_locomo"}
SDIR = {"lme": f"{ROOT}/states/cocomem", "locomo": f"{ROOT}/states/locomo_cocomem"}

WITNESS = """You maintain a memory {kind} of a personal assistant. It was written from the facts below.

Facts:
{facts}

The {kind}, sentence by sentence:
{unit}

For every sentence, list the ids of ALL facts it was derived from: the facts whose content it states or paraphrases, and every fact that it counts, adds up, compares, ranks, orders in time, measures a duration from, or draws a conclusion from. A sentence about a total, a count, a maximum or minimum, a first or last event, a duration or a trend depends on every fact that enters that total, count, extreme, order, duration or trend, including facts it does not name. Write "none" if the sentence is derived from none of these facts.

Output exactly one line per sentence, in the form "<k>: <ids separated by commas>", and nothing else."""


def unit_sents(u):
    from cocomem import split_sentences
    t = u["text_cited"]
    return [t[a:b].strip() for a, b in split_sentences(t)]


def _parse(r, n, allowed):
    from cocomem import cited_ids
    out = [[] for _ in range(n)]
    for line in r.splitlines():
        m = re.match(r"\s*\(?(\d+)\)?\s*[:.)]\s*(.*)$", line)
        if not m or int(m.group(1)) >= n:
            continue
        # The annotator may compact a list as c12-c16, just like inline citations.
        # Keep all operands in such ranges instead of silently retaining only endpoints.
        ids = [cid for chunk in re.findall(r"c\d+(?:\s*[,–—-]\s*c?\d+)*", m.group(2))
               for cid in cited_ids(f"[{chunk}]" )]
        out[int(m.group(1))] = [c for c in dict.fromkeys(ids) if c in allowed]
    return out


async def annotate_unit(llm, m, u, usage=None):
    """Witness of every sentence of unit u: the compressor's own account of what each sentence was derived from,
    united with the sources the sentence already cites (the compressor's declared sources).  Restricted to the unit's
    anchored inputs, so the witness is a sub-view of the unit's derivation and is anchored with it."""
    from cocomem import cited_ids
    sents = unit_sents(u)
    ins = [c for c in u["inputs"] if c in m.claims]
    if not sents:
        return sents, []
    allowed = set(ins)
    shown = "\n".join(f"({k}) {s}" for k, s in enumerate(sents))
    kind = {"note": "note", "digest": "digest", "profile": "profile"}[u["level"]]
    r = await llm.chat(WITNESS.format(kind=kind, facts=m._fact_lines(ins), unit=shown),
                       max_tokens=min(60 + 48 * len(sents) + 4 * len(ins), 4000), usage=usage)
    W = _parse(r, len(sents), allowed)
    W = [list(dict.fromkeys([c for c in cited_ids(s) if c in allowed] + w)) for s, w in zip(sents, W)]
    return sents, W


async def annotate_mem(llm, m, usage=None):
    uids = list(m.units)
    got = await asyncio.gather(*(annotate_unit(llm, m, m.units[x], usage) for x in uids))
    return {x: {"sents": s, "witness": w, "text_cited": m.units[x]["text_cited"]} for x, (s, w) in zip(uids, got)}


def attach(m, path):
    """Attach a precomputed witness to a loaded memory (only to units whose text is unchanged since annotation)."""
    if not os.path.exists(path):
        return 0
    d = json.load(open(path))["units"]
    n = 0
    for uid, v in d.items():
        u = m.units.get(uid)
        if u is not None and u["text_cited"] == v["text_cited"] and len(v["sents"]) == len(v["witness"]):
            u["sents"], u["witness"] = list(v["sents"]), [list(w) for w in v["witness"]]
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
    os.makedirs(WDIR[a.kind], exist_ok=True)
    files = state_files(a.kind, a.required_only)
    sem = asyncio.Semaphore(a.inst)

    async def one(f):
        out = f"{WDIR[a.kind]}/{os.path.basename(f)}"
        if os.path.exists(out):
            return "cached"
        async with sem:
            m = CoCoMem.from_json(llm, json.load(open(f)))
            u = Usage()
            units = await annotate_mem(llm, m, u)
            json.dump({"units": units, "usage": u.as_dict(), "n_claims": len(m.claims)}, open(out + ".tmp", "w"))
            os.replace(out + ".tmp", out)
            return "ok"

    res = await gather_limited([one(f) for f in files], desc=f"witness-{a.kind}", every=20)
    print(a.kind, {r: res.count(r) for r in set(res)}, flush=True)


def cmd_stats(a):
    """Witness quality and size.  Recall: share of held-out verified positives (sentence k changes under the
    correction of claim c) with c in W_k; compared with the compressor's inline citations.  Fan-in |W_k| by level,
    and its correlation with the size of the user's ledger (the cost of a repair grows with the fan-in, not with
    the history)."""
    import numpy as np
    E = [json.loads(l) for l in open(a.pairs)]
    E = [e for e in E if not e.get("rt")]
    rec = {}
    for e in E:
        if not e["y"]:
            continue
        cid = re.match(r"\[(c\d+)\]", e["old"]).group(1)
        for lev in (e["level"], "all"):
            r = rec.setdefault(lev, [0, 0, 0])
            r[0] += 1
            r[1] += cid in (e.get("W") or [])
            r[2] += cid in set(re.findall(r"c\d+", " ".join(re.findall(r"\[([^\]]*)\]", e["sent"]))))
    out = {"recall": {k: {"n": v[0], "witness": 100 * v[1] / v[0], "citations": 100 * v[2] / v[0]} for k, v in rec.items()}}
    fan, size, wtok = {}, [], []
    required = {os.path.basename(f) for f in state_files("lme", True)} if a.required_only else None
    for f in glob.glob(f"{WDIR['lme']}/*.json"):
        if required is not None and os.path.basename(f) not in required:
            continue
        d = json.load(open(f))
        per = []
        for uid, v in d["units"].items():
            lev = "profile" if uid.startswith("profile") else "digest" if uid.startswith("d") else "note"
            for w in v["witness"]:
                fan.setdefault(lev, []).append(len(w))
                per.append(len(w))
        if per:
            size.append((d.get("n_claims", 0), float(np.mean(per)), max(per)))
        wtok.append(d["usage"]["in"] + d["usage"]["out"])
    S = np.array(size, float)
    q = np.quantile(S[:, 0], [0, .25, .5, .75, 1])
    out["fanin"] = {k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p90": float(np.quantile(v, .9)),
                        "max": int(max(v))} for k, v in fan.items()}
    out["fanin_vs_history"] = {"r_mean": float(np.corrcoef(S[:, 0], S[:, 1])[0, 1]),
                               "by_quartile": [float(S[(S[:, 0] >= q[i]) & (S[:, 0] <= q[i + 1]), 1].mean()) for i in range(4)],
                               "edges": [float(x) for x in q]}
    out["write_tokens_per_memory"] = float(np.mean(wtok)) if wtok else 0.0
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--kind", default="lme")
    ap.add_argument("--conc", type=int, default=96)
    ap.add_argument("--inst", type=int, default=12)
    ap.add_argument("--required-only", action="store_true", help="Only v20 training/evaluation users and the paper test split")
    ap.add_argument("--pairs", default=f"{ROOT}/models/tnet20/test.jsonl")
    ap.add_argument("--out", default="witness_stats.json")
    a = ap.parse_args()
    if a.cmd == "annotate":
        asyncio.run(cmd_annotate(a))
    elif a.cmd == "stats":
        cmd_stats(a)
    elif a.cmd == "count":
        print(len(state_files(a.kind, a.required_only)))
