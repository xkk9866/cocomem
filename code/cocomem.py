"""CoCoMem: correction-commutative memory compression.

Memory = derivation DAG over an addressable, versioned claim ledger:
  L0 claims (extracted per session)  ->  L1 session notes, L2 topic digests, L3 user profile
Every unit is a function of an anchored set of input claims, and carries a lineage
(the subset of inputs it causally depends on).  A correction is an intervention on
claims: ground -> version -> lineage closure -> local decompress/revise/recompress.
"""
import asyncio
import collections
import copy
import os
import re

import numpy as np

import prompts
from claims import parse_claims
from data import format_session
from embed import Embedder
from llm import Usage, parse_json

CITE = re.compile(r"\[\s*(c\d+(?:\s*[,–—-]\s*c?\d+)*)\s*\]")

# Compressor mode.  "text": the compressor writes the unit text (v1-v21; derivation records, if any, are attributed
# afterwards by a second pass).  "plan": plan-first, self-maintainable compression (v22): the compressor writes
# every sentence together with the derivation it is stated from (prompts.PLAN_*); the derivation lines are the
# materialized auxiliary view of the unit, the sentence lines form its text.
COMPRESSOR = os.environ.get("CM_COMPRESSOR", "text")
# Derived levels.  The record-first memory has two: session notes and topic digests, each written from a bounded
# set of claims (one session; at most 40 claims of one topic).  The text memories (v1-v21) also kept one profile per
# owner, written from every claim of the owner.  Read with the same reader, the v21 memories answer 62.2% of the
# original questions with the profile in the context and 63.2% without it (n=201, 13 gained, 11 lost), while the
# profile lies in the closure of three corrections out of four.
PROFILE_LEVEL = COMPRESSOR != "plan"
INTAKE = os.environ.get("CM_INTAKE", "resolve")      # how an arriving session is entered into the ledger (CoCoMem.resolve)
REVISED_TAG = INTAKE == "ground"                     # the earlier read path marked maintained units "(revised <date>)"
FULL_WINDOW = 24576                     # context window of the 8B compressor (servers on crowded cards may get less)
_RID = re.compile(r"\bc\d+\b")
_BARE_ID = re.compile(r"(?<![A-Za-z0-9])(\d+)(\s*=)")
_PLAN_LINE = re.compile(r"^\s*(?:[-*\d.)\s]*)?\**\s*([TDS])\s*\**\s*:\s*(.*)$", re.I)


def restrict_record(rec, allowed):
    """Keep a derivation record's entries whose ids are inputs of the unit; a bare numeric id ("237 = ...") is read
    as the claim id it abbreviates when that claim is an input.  Entries without an id (computations) are kept."""
    if not rec or rec.strip().lower() == "none":
        return "none"
    rec = _BARE_ID.sub(lambda m: f"c{m.group(1)}{m.group(2)}" if f"c{m.group(1)}" in allowed else m.group(0), rec)
    parts = [p.strip() for p in rec.split(";") if p.strip()]
    kept = [p for p in parts if not _RID.search(p) or any(c in allowed for c in _RID.findall(p))]
    return "; ".join(kept) if kept else "none"


def _own_entries(rec, sent):
    """One record line written for several sentences: the entries that belong to `sent` are those of the facts it
    cites, plus the entries without an id (computations).  A sentence that cites none of the record's facts keeps the
    whole record, so the split never drops a dependency."""
    cid = set(cited_ids(sent))
    parts = [p.strip() for p in rec.split(";") if p.strip()]
    if not any(set(_RID.findall(p)) & cid for p in parts):
        return rec
    return "; ".join(p for p in parts if not _RID.search(p) or set(_RID.findall(p)) & cid)


def parse_plan(out, allowed):
    """Record-first output -> (sentences, records, ids, text_cited, ok).  T:/D:/S: triples: the T line (what the
    sentence will state) conditions the record and is not stored; a sentence without a record line gets "none".
    ok is False when the output holds no S: line at all (the compressor left the grammar); the caller then writes the
    unit as prose (CoCoMem._compress) -- the raw output is never used as unit text."""
    sents, recs, pend, last = [], [], None, None
    for line in out.splitlines():
        m = _PLAN_LINE.match(line)
        if not m:
            if last == "S" and line.strip():
                sents[-1] = (sents[-1] + " " + line.strip()).strip()          # a wrapped sentence line
            continue
        tag, body = m.group(1).upper(), m.group(2).strip().strip('"')
        last = tag
        if tag == "T":
            pend = None                                # a new triple: a record left without its sentence is dropped
        elif tag == "D":
            pend = body or "none"
        else:
            if not body:
                last = None
                continue
            sents.append(body)
            recs.append(pend if pend is not None else "none")
            pend = None
    ok = bool(sents)
    recs = [restrict_record(r, allowed) for r in recs]
    if ok:      # an S line that holds several sentences: each keeps the entries of the line's record that are its own
        s2, r2 = [], []
        for s, r in zip(sents, recs):
            parts = [s[a:b].strip() for a, b in split_sentences_v2(s)]
            parts = [p for p in parts if p] or [s]
            s2 += parts
            r2 += [_own_entries(r, p) for p in parts] if len(parts) > 1 else [r]
        sents, recs = s2, r2
    ids = [sorted(((set(_RID.findall(r)) | {c for c in cited_ids(s)}) & set(allowed)), key=lambda c: int(c[1:]))
           for r, s in zip(recs, sents)]
    return sents, recs, ids, " ".join(sents), ok


def strip_cites(t):
    return CITE.sub("", t).replace(" .", ".").strip()


def cited_ids(t):
    out = []
    for m in CITE.finditer(t):
        for part in m.group(1).split(","):
            ends = re.split(r"\s*[–—-]\s*", part.strip())
            if len(ends) == 2:
                a, b = int(ends[0].lstrip("c")), int(ends[1].lstrip("c"))
                if 0 <= b - a <= 60:
                    out += [f"c{i}" for i in range(a, b + 1)]
                    continue
            out += [e if e.startswith("c") else "c" + e for e in ends]
    return list(dict.fromkeys(out))


def ntok(t):
    return len(t) // 4 + 1


_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("/home/yaokang/models/Qwen3-8B-AWQ")
    return _TOK


def ntok_exact(t):
    return len(_tok()(t, add_special_tokens=False).input_ids) if t else 0


def chat_prefix(prompt):
    return _tok().apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)


def first_sentence(t):
    t = t.lstrip()
    r = split_sentences(t)
    return t[r[0][0]:r[0][1]].strip() if r else ""


_CITE_WS = re.compile(r"\s*\[\s*(c\d+(?:\s*[,–—-]\s*c?\d+)*)\s*\]")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(t):
    """Sentence char ranges of a (cited) unit text."""
    out, pos = [], 0
    for m in _SENT.finditer(t):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(t)))
    return [(a, b) for a, b in out if t[a:b].strip()]


_ABBR = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Mx|St|Jr|Sr|Prof|Gen|Col|Sgt|Lt|Capt|Rev|Hon|Pres|Gov|Sen|Rep|Dept|Univ|Inc|Ltd|Co|"
                   r"Corp|Bros|Mt|Ft|Ave|Rd|Blvd|No|Nos|Vol|Fig|approx|vs|etc|al|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|"
                   r"Nov|Dec|U\.S|U\.K|e\.g|i\.e|a\.m|p\.m)\.$")
_CITES_ONLY = re.compile(r"(?:\[\s*c\d[^\]]*\]\s*(?:[.,;]\s*)?)+")
_HEADER = re.compile(r"(?:\*\*)?[A-Za-z][A-Za-z' /&-]{0,30}:(?:\*\*)?:?")


def split_sentences_v2(t):
    """Sentence char ranges of a (cited) unit text, robust to abbreviations ("Dr."), headers ("**Digest:**") and
    citation-only fragments, which the plain splitter turns into spurious 'sentences'.  Used by the v21 derivation
    records and the derivation-conditioned commutator; the older strategies keep split_sentences."""
    out, pos = [], 0
    for m in _SENT.finditer(t):
        if "\n" not in m.group(0) and _ABBR.search(t[pos:m.start()].rstrip()):
            continue                                   # "Dr. Johnson" is not a sentence boundary
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(t)))
    out = [(a, b) for a, b in out if t[a:b].strip()]
    merged = []
    for a, b in out:
        s = t[a:b].strip()
        if merged and (_CITES_ONLY.fullmatch(s) or len(s) < 4):
            merged[-1] = (merged[-1][0], b)            # "[c15][c16]" or "." belongs to the previous sentence
        else:
            merged.append((a, b))
    res, i = [], 0
    while i < len(merged):
        a, b = merged[i]
        if _HEADER.fullmatch(t[a:b].strip()) and i + 1 < len(merged):
            res.append((a, merged[i + 1][1]))          # "**Digest:**" is the head of the next sentence
            i += 2
            continue
        res.append((a, b))
        i += 1
    return res


def sentence_ranges(text_cited):
    """Sentence ranges expressed in the citation-stripped text (the coordinates of the lineage field)."""
    keep = np.ones(len(text_cited), dtype=bool)
    for m in _CITE_WS.finditer(text_cited):
        keep[m.start():m.end()] = False
    mp = np.cumsum(keep) - 1
    res = []
    for a, b in split_sentences(text_cited):
        ks = [int(mp[i]) for i in range(a, b) if keep[i]]
        res.append((min(ks), max(ks) + 1) if ks else (0, 0))
    return res


async def emb(texts, query=False):
    return await asyncio.to_thread(Embedder.get().encode, list(texts), query)


class CoCoMem:
    """One memory instance (one user / one LongMemEval question)."""

    def __init__(self, llm, note_words=100, digest_words=150, profile_words=250, cluster_dist=0.30, persons=None):
        self.llm = llm
        self.persons = persons   # None: user-assistant chat; [A, B]: dialog between two named people
        self.text_words = {"note": note_words, "digest": digest_words, "profile": profile_words}   # prose prompts
        if COMPRESSOR == "plan" and (note_words, digest_words, profile_words) == (100, 150, 250):
            # the text compressor overshoots its budgets (measured means: notes 120, digests 105, profiles 500+ words);
            # the record-first prompts are given the lengths the text compressor produces, so that the two write
            # paths are compared at the same unit size
            note_words, digest_words, profile_words = 120, 200, 600
        self.note_words, self.digest_words, self.profile_words = note_words, digest_words, profile_words
        self.cluster_dist = cluster_dist
        self.claims = {}   # cid -> claim
        self.units = {}    # uid -> unit
        self.usage = {"write": Usage(), "repair": Usage(), "read": Usage()}
        self.n = 0

    # ------------------------------------------------------------------ write
    def _new_cid(self):
        self.n += 1
        return f"c{self.n}"

    async def extract(self, session, usage):
        if self.persons:
            a, b = self.persons
            text = await self.llm.chat(prompts.EXTRACT_DIALOG.format(a=a, b=b, session=format_session(session["turns"], session["date"])),
                                       max_tokens=900, usage=usage)
            return parse_claims(text, codes={"A": a, "B": b})
        text = await self.llm.chat(prompts.EXTRACT.format(session=format_session(session["turns"], session["date"])),
                                   max_tokens=900, usage=usage)
        return parse_claims(text)

    def _fact_lines(self, cids, show_status=False):
        lines = []
        for c in cids:
            x = self.claims[c]
            tag = f"[{c}] ({x['date'][:10]}, {x['speaker']})"
            if show_status and x["status"] == "superseded":
                tag += " [SUPERSEDED]"
            lines.append(f"{tag} {x['text']}")
        return "\n".join(lines)

    async def add_session(self, session, usage=None, make_note=True):
        usage = usage or self.usage["write"]
        raw = await self.extract(session, usage)
        new = []
        for r in raw:
            cid = self._new_cid()
            self.claims[cid] = {"id": cid, "text": r["text"], "speaker": r["speaker"], "turn": r["turn"],
                                "sid": session["sid"], "date": session["date"], "status": "current",
                                "superseded_by": [], "superseded_at": None, "revises": [],
                                "revises_hint": r["revises"]}
            new.append(cid)
        if make_note and new:
            await self._make_note(session, new, usage)
        return new

    # ---- compression of one unit in the configured compressor mode -------------------------------------------
    # output windows.  The word budget of the prompt governs the length of a unit; the window only has to hold a
    # well-formed output, which in record-first mode is three lines per sentence (the first build cut 800 / 1200 /
    # 1800 tokens and truncated units inside their records).  A unit that still ends at the window is not kept.
    PLAN_TOKENS = {"note": 2000, "digest": 1600, "profile": 6000}
    TEXT_TOKENS = {"note": 400, "digest": 600, "profile": 900}

    # A record-first unit is written sentence by sentence, and a compressor that writes one triple per sentence tends
    # to spend one sentence per fact (first sample: notes of 14 sentences / 223 words against a 120-word budget).  The
    # word budget is therefore also given as the number of sentences it holds at the compressor's own mean sentence
    # length at that level (measured on record-first samples: 17, 21 and 30 words).
    SENT_WORDS = {"note": 17, "digest": 21, "profile": 30}

    def _unit_prompt_for(self, level, cids, date=None, person=None, plan=None):
        facts = self._fact_lines(cids)
        plan = COMPRESSOR == "plan" if plan is None else plan
        words = {"note": self.note_words, "digest": self.digest_words, "profile": self.profile_words}[level] if plan \
            else self.text_words[level]
        kw = {"words": words, "facts": facts}
        if plan:
            kw["sents"] = max(3, words // self.SENT_WORDS[level])
        if level == "note":
            return (prompts.PLAN_NOTE if plan else prompts.NOTE).format(date=date, **kw)
        if level == "digest":
            return (prompts.PLAN_DIGEST if plan else prompts.DIGEST).format(**kw)
        if person:
            return (prompts.PLAN_PROFILE_PERSON if plan else prompts.PROFILE_PERSON).format(person=person, **kw)
        return (prompts.PLAN_PROFILE if plan else PROFILE).format(**kw)

    async def _compress(self, level, cids, usage, date=None, person=None):
        """Write one unit from its inputs.  Returns the fields of the unit that depend on the text: in plan mode
        the sentences come with their derivation records (the auxiliary view), in text mode the text alone."""
        async def prose():
            out = await self.llm.chat(self._unit_prompt_for(level, cids, date=date, person=person, plan=False),
                                      max_tokens=self.TEXT_TOKENS[level], usage=usage)
            return {"text_cited": out.strip(), "text": strip_cites(out), "cited": [c for c in cited_ids(out) if c in cids]}

        if COMPRESSOR != "plan":
            return await prose()
        from llm import ContextOverflow
        one, ok = Usage(), False
        try:
            out = await self.llm.chat(self._unit_prompt_for(level, cids, date=date, person=person, plan=True),
                                      max_tokens=self.PLAN_TOKENS[level], usage=one)
            sents, recs, ids, text_cited, ok = parse_plan(out, set(cids))
        except ContextOverflow:
            # on a server with a reduced window the caller defers the unit to a full-window server; on the full
            # window the record-first output of this unit does not fit next to its inputs
            if int(os.environ.get("CM_MAXLEN", "0") or FULL_WINDOW) < FULL_WINDOW:
                raise
        if usage is not None:
            usage += one
        if ok and one.completion_tokens < self.PLAN_TOKENS[level]:
            return {"text_cited": text_cited, "text": strip_cites(text_cited),
                    "cited": [c for c in cited_ids(text_cited) if c in cids],
                    "sents": sents, "deriv": recs, "dids": ids, "plan_ok": True}
        # The compressor left the grammar (no sentence line), did not finish inside the output window, or the output
        # does not fit the context.  Such a unit is written as prose and its sentences carry no record: the commutator
        # then sees "none", as it does for any sentence without a record.  plan_ok = False marks the unit; the share
        # is reported with the write statistics.
        w = await prose()
        t = w["text_cited"]
        sents = [x for x in (t[a:b].strip() for a, b in split_sentences_v2(t)) if x]
        allowed = set(cids)
        return {**w, "sents": sents, "deriv": ["none"] * len(sents),
                "dids": [sorted(set(cited_ids(s)) & allowed, key=lambda c: int(c[1:])) for s in sents], "plan_ok": False}

    async def _make_note(self, session, cids, usage):
        uid = f"n_{session['sid']}"
        w = await self._compress("note", cids, usage, date=session["date"])
        self.units[uid] = {"id": uid, "level": "note", "inputs": list(cids), "date": session["date"], "lineage": {}, **w}

    async def consolidate(self, usage=None):
        """Sleep-time consolidation: topic digests over user claims + user profile."""
        usage = usage or self.usage["write"]
        owners = self.persons or ["user"]
        user = [c for c, x in self.claims.items() if x["speaker"] in owners and x["status"] == "current"]
        tasks = []
        if len(user) >= 3:
            from sklearn.cluster import AgglomerativeClustering
            E = await emb([self.claims[c]["text"] for c in user])
            lab = AgglomerativeClustering(n_clusters=None, distance_threshold=self.cluster_dist, metric="cosine",
                                          linkage="average").fit_predict(E)
            groups = {}
            for c, l in zip(user, lab):
                groups.setdefault(int(l), []).append(c)
            k = 0
            for g in groups.values():
                if len(g) < 3 or len({self.claims[c]["sid"] for c in g}) < 2:
                    continue
                g = sorted(g, key=lambda c: (self.claims[c]["date"], int(c[1:])))
                for i in range(0, len(g), 40):
                    chunk = g[i:i + 40]
                    if len(chunk) >= 3:
                        tasks.append(self._make_digest(f"d{k}", chunk, usage))
                        k += 1
        if PROFILE_LEVEL and self.persons:
            for k, person in enumerate(self.persons):
                tasks.append(self._make_profile([c for c in user if self.claims[c]["speaker"] == person], usage,
                                                uid=f"profile{k}", person=person))
        elif PROFILE_LEVEL:
            tasks.append(self._make_profile(user, usage))
        await asyncio.gather(*tasks)

    async def _make_digest(self, uid, cids, usage):
        w = await self._compress("digest", cids, usage)
        self.units[uid] = {"id": uid, "level": "digest", "inputs": list(cids), "date": self.claims[cids[-1]]["date"],
                           "lineage": {}, **w}

    async def _make_profile(self, cids, usage, uid="profile", person=None):
        cids = sorted(cids, key=lambda c: (self.claims[c]["date"], int(c[1:])))
        if not cids:
            return
        w = await self._compress("profile", cids, usage, person=person)
        self.units[uid] = {"id": uid, "level": "profile", "person": person, "inputs": list(cids),
                           "date": self.claims[cids[-1]]["date"], "lineage": {}, **w}

    def set_lineage(self, mode, cle=None, thr=0.5):
        """Lineage of every unit: coarse = all inputs; self = LLM citations; clf = counterfactual lineage field."""
        if mode in ("clf", "ncf"):
            self.clf = cle
        for u in self.units.values():
            self.unit_lineage(u, mode, cle, thr)

    def unit_lineage(self, u, mode, clf=None, thr=0.5):
        if mode == "coarse":
            u["lineage"][mode] = list(u["inputs"])
        elif mode == "self":
            u["lineage"][mode] = [c for c in cited_ids(u["text_cited"]) if c in self.claims]
            t = u["text_cited"]
            u["support"] = {}
            for k, (a, b) in enumerate(split_sentences(t)):
                for c in cited_ids(t[a:b]):
                    u["support"].setdefault(c, []).append(k)
        elif mode == "ncf":
            u["lineage"][mode], u["support"] = [], {}
            ins = [c for c in u["inputs"] if c in self.claims]
            if not ins:
                return
            F = clf.field(u["text_cited"], [f"[{c}] {self.claims[c]['text']}" for c in ins], level=u["level"])
            for c, row in zip(ins, F):
                ks = [int(k) for k in np.nonzero(row >= thr)[0]]
                if ks:
                    u["lineage"][mode].append(c)
                    u["support"][c] = ks
        elif mode == "clf":
            u["lineage"][mode], u["support"] = [], {}
            ins = [c for c in u["inputs"] if c in self.claims]
            if not ins:
                return
            fields, scores = clf.field(u["text_cited"], [self.claims[c]["text"] for c in ins], level=u["level"])
            sents = sentence_ranges(u["text_cited"])
            for c, f, s in zip(ins, fields, scores):
                if s >= thr:
                    u["lineage"][mode].append(c)
                    u["support"][c] = sorted({k for k, (a, b) in enumerate(sents)
                                              for (ts, te, p) in f if p >= thr and ts < b and te > a})

    # ------------------------------------------------------------ correction
    async def ground(self, new_cids, date, usage, k=8, context=""):
        """Map new claims to the stored claims they revise (embedding candidates + LLM verification).
        Candidates are retrieved with the new claim and with its REVISES hint; the verifier also sees the
        user's own words in the correction session."""
        pool = [c for c, x in self.claims.items() if x["status"] == "current" and c not in new_cids]
        if not pool:
            return {}
        P = await emb([self.claims[c]["text"] for c in pool])
        qs = [self.claims[c]["text"] for c in new_cids] + [self.claims[c].get("revises_hint") or self.claims[c]["text"]
                                                            for c in new_cids]
        Q = await emb(qs, query=True)
        n = len(new_cids)
        out = {}

        async def one(i, c):
            idx = list(dict.fromkeys(list(np.argsort(-(Q[i] @ P.T))[:k]) + list(np.argsort(-(Q[n + i] @ P.T))[:k])))
            cands = [pool[j] for j in idx]
            hint = self.claims[c].get("revises_hint")
            r = await self.llm.chat(prompts.GROUND.format(
                date=date, claim=self.claims[c]["text"], context=context[:1500],
                revises_hint=(f'(The user indicated it revises: "{hint}")\n' if hint else ""),
                cands=self._fact_lines(cands)), max_tokens=200, usage=usage)
            js = parse_json(r, {}) or {}
            rev = js.get("revised", []) if isinstance(js, dict) else []
            rev = [x for x in rev if isinstance(x, str) and x in cands]
            if rev:
                out[c] = rev

        await asyncio.gather(*(one(i, c) for i, c in enumerate(new_cids)))
        return out

    async def resolve(self, session, new_cids, usage, k=8, limit=24):
        """Read a session against the ledger.  A correction is elliptical ("it was 8 months, not a year"): it names
        neither the slot it revises nor its subject, so a claim extracted from the session alone cannot carry it.
        The stored claims nearest to what was said (queries: the sentences of the session, the claims extracted from
        it and their REVISES hints) are shown to the compressor together with the session, and one call returns, for
        every stored claim the session revises, the claim as it reads now: a sentence in the wording of the slot it
        replaces, with the corrected values.  Each revision is entered as the next version of its slot, so the
        correction delta = (c -> c') is an edit of a ledger entry and the slot keeps the date of its first record.
        Returns {new claim id: [revised claim id]}."""
        pool = [c for c, x in self.claims.items() if x["status"] == "current" and c not in new_cids]
        said = [(i, t) for i, t in enumerate(session["turns"]) if self.persons or t["role"] == "user"]
        if not pool or not said:
            return {}
        qs = []
        for _, t in said:
            qs += [t["content"][a:b].strip() for a, b in split_sentences_v2(t["content"])]
        for c in new_cids:
            qs += [self.claims[c]["text"], self.claims[c].get("revises_hint") or ""]
        qs = [q for q in dict.fromkeys(qs) if len(q) >= 8]
        P = await emb([self.claims[c]["text"] for c in pool])
        S = (await emb(qs, query=True)) @ P.T
        best = {}
        for i in range(len(qs)):
            for j in np.argsort(-S[i])[:k]:
                best[int(j)] = max(best.get(int(j), -1.0), float(S[i, j]))
        cands = sorted((pool[j] for j in sorted(best, key=lambda j: -best[j])[:limit]),
                       key=lambda c: (self.claims[c]["date"], int(c[1:])))
        if self.persons:
            owner = f"an assistant that follows the conversations between two friends, {self.persons[0]} and {self.persons[1]}"
            ctx = "\n".join(f"{t['role']}: {t['content'].strip()}" for _, t in said)
        else:
            owner = "a personal chat assistant"
            ctx = "\n".join(f"User: {t['content'].strip()}" for _, t in said)
        r = await self.llm.chat(prompts.RESOLVE.format(owner=owner, date=session["date"], context=ctx[:3000],
                                                       cands=self._fact_lines(cands)), max_tokens=500, usage=usage)
        js = parse_json(r, {}) or {}
        rows = js.get("revised", []) if isinstance(js, dict) else []
        out, seen = {}, set()
        for x in rows if isinstance(rows, list) else []:
            if not isinstance(x, dict) or not isinstance(x.get("now"), str):
                continue
            old, now = str(x.get("id", "")).strip().strip("[]").strip().lower(), x["now"].strip()
            if old.isdigit():
                old = "c" + old                        # the id returned without its prefix ("292" for [c292])
            if old not in cands or old in seen or len(now) < 8:
                continue
            o = self.claims[old]
            if re.sub(r"[^a-z0-9]", "", now.lower()) == re.sub(r"[^a-z0-9]", "", o["text"].lower()):
                continue                               # the stored claim returned unchanged: nothing is revised
            seen.add(old)
            cid = self._new_cid()
            self.claims[cid] = {"id": cid, "text": now, "speaker": o["speaker"], "turn": said[0][0], "sid": session["sid"],
                                "date": session["date"], "since": o.get("since") or o["date"], "status": "current",
                                "superseded_by": [], "superseded_at": None, "revises": [], "revises_hint": None}
            out[cid] = [old]
        return out

    def apply_versions(self, revmap, date, keep_history=True):
        revised = set()
        for new, olds in revmap.items():
            self.claims[new]["revises"] = list(olds)
            if not REVISED_TAG and olds and olds[0] in self.claims:
                self.claims[new].setdefault("since", self.claims[olds[0]].get("since") or self.claims[olds[0]]["date"])
            for o in olds:
                revised.add(o)
                if keep_history:
                    self.claims[o]["status"] = "superseded"
                    self.claims[o]["superseded_by"].append(new)
                    self.claims[o]["superseded_at"] = date
        if not keep_history:
            for o in revised:
                self.claims.pop(o, None)
        return revised

    def closure(self, revised, mode):
        return [u for u in self.units.values() if set(u["lineage"].get(mode, [])) & revised]

    def _diffs(self, u, revmap_inv, lineage_mode):
        diffs, news_ids = [], []
        for old, news in revmap_inv.items():
            if old in u["lineage"].get(lineage_mode, []):
                oldtxt = self.claims[old]["text"] if old in self.claims else self._removed.get(old, "")
                for nw in news:
                    diffs.append(f"[{old}] {oldtxt}  ->  [{nw}] {self.claims[nw]['text']}")
                    news_ids.append(nw)
        return diffs, news_ids

    async def repair_unit(self, u, revmap_inv, date, lineage_mode, usage, with_facts=True, span_local=False):
        """Local decompression (lineage facts) -> revision -> recompression of one unit.
        span_local: only the sentences in the lineage field's support are regenerated; the rest of
        the unit is copied verbatim (commutativity on the complement holds by construction)."""
        diffs, news_ids = self._diffs(u, revmap_inv, lineage_mode)
        if not diffs:
            return False
        lin = [c for c in u["lineage"].get(lineage_mode, []) if c in self.claims and self.claims[c]["status"] == "current"]
        facts = self._fact_lines(list(dict.fromkeys(lin + news_ids))) if with_facts else "(not available)"
        kind = {"note": "note", "digest": "digest", "profile": "profile"}[u["level"]]
        old_text = u["text_cited"]
        if span_local:
            sents = split_sentences(old_text)
            marked = sorted({k for old in revmap_inv for k in u.get("support", {}).get(old, []) if k < len(sents)})
            if not marked:
                marked = list(range(len(sents)))
            shown = "\n".join((f"<<{k}>> " if k in marked else f"({k}) ") + old_text[a:b].strip()
                              for k, (a, b) in enumerate(sents))
            r = await self.llm.chat(prompts.REPAIR_SPAN.format(kind=kind, unit=shown, date=date, diffs="\n".join(diffs),
                                                               facts=facts, ids=", ".join(str(k) for k in marked)),
                                    max_tokens=200 + 120 * len(marked), usage=usage)
            js = {}
            for line in r.splitlines():
                m = re.match(r"\s*<{0,2}\(?(\d+)\)?>{0,2}\s*[:.)]\s*(.*)$", line)
                if m and int(m.group(1)) in marked:
                    js[m.group(1)] = m.group(2).strip().strip('"')
            if not js:
                return False
            pieces, changed = [], False
            for k, (a, b) in enumerate(sents):
                seg = old_text[a:b].strip()
                if k in marked and str(k) in js:
                    new_seg = js[str(k)]
                    changed |= new_seg != seg
                    seg = new_seg
                if seg:
                    pieces.append(seg)
            if not changed:
                return False
            new_text = " ".join(pieces)
        else:
            r = await self.llm.chat(prompts.REPAIR.format(kind=kind, unit=old_text, date=date,
                                                          diffs="\n".join(diffs), facts=facts),
                                    max_tokens=1000 if u["level"] == "profile" else 600, usage=usage)
            new_text = re.sub(r"^```\w*|```$", "", r.strip()).strip().strip('"').strip()
            if not new_text or new_text.upper().startswith("UNCHANGED") or new_text == old_text:
                return False
        u["text_cited"] = new_text
        u["text"] = strip_cites(new_text)
        # anchored grouping: the new claim takes the slot of the claim it revises
        for old, news in revmap_inv.items():
            if old in u["inputs"]:
                u["inputs"] = [c for c in u["inputs"] if c != old] + [n for n in news if n not in u["inputs"]]
            for m in u["lineage"]:
                if old in u["lineage"][m]:
                    u["lineage"][m] = [c for c in u["lineage"][m] if c != old] + [n for n in news if n not in u["lineage"][m]]
        u["revised_at"] = date
        return True

    async def correct(self, session, strategy="cle", usage=None):
        """Process a new session that may contain corrections, using a given maintenance strategy."""
        self._edits = {}
        info = await self._correct(session, strategy, usage)
        info["edits"] = self._edits          # {new claim: [revised claim, its text, the claim as it reads now]}
        return info

    async def _correct(self, session, strategy, usage):
        usage = usage or self.usage["repair"]
        self._removed = {}
        date = session["date"]
        if strategy == "append":
            await self.add_session(session, usage=usage)
            return {"revised": [], "repaired": []}
        cfg = STRATEGIES.get(strategy, {})
        # intake: "resolve" reads the session against the ledger (resolve); "ground" is the earlier two-step intake
        # (claims extracted from the session alone, then verified against retrieved candidates), kept as an ablation
        blind = strategy in ("simupdate", "scan") or cfg.get("intake", INTAKE) == "ground"
        new = await self.add_session(session, usage=usage, make_note=blind)
        if strategy in ("simupdate", "scan"):
            return await self._correct_unit_level(new, date, strategy, usage)
        if blind:
            ctx = "\n".join(t["content"] for t in session["turns"] if t["role"] == "user")
            revmap = await self.ground(new, date, usage, context=ctx)
        else:
            revmap = await self.resolve(session, new, usage)
            if new or revmap:                # the note of the session states the claims as they read now
                await self._make_note(session, new + list(revmap), usage)
        self._edits = {n: [olds[0], self.claims[olds[0]]["text"], self.claims[n]["text"]] for n, olds in revmap.items()}
        inv = {}
        for nw, olds in revmap.items():
            for o in olds:
                inv.setdefault(o, []).append(nw)
        if not cfg["history"]:
            self._removed = {o: self.claims[o]["text"] for o in inv}
        revised = set(inv)
        if strategy == "rebuild":
            self.apply_versions(revmap, date, keep_history=True)
            await self._rebuild(usage)
            return {"revised": sorted(revised), "repaired": ["*"]}
        if cfg.get("repair") == "none":      # version slots only: ground + supersede, derived units untouched
            self.apply_versions(revmap, date, keep_history=True)
            return {"revised": sorted(revised), "repaired": [], "closure": []}
        if cfg.get("repair") == "ccf":
            return await self._ccf_repair(revised, inv, revmap, date, cfg, usage)
        if cfg.get("repair") == "comm":
            return await self._comm_repair(revised, inv, revmap, date, cfg, usage)
        if cfg.get("repair") == "witness":
            return await self._witness_repair(revised, inv, revmap, date, cfg, usage)
        if cfg.get("repair") == "deriv":
            return await self._deriv_repair(revised, inv, revmap, date, cfg, usage)
        mode = cfg["lineage"]
        targets = self.closure(revised, mode)
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        if cfg.get("repair") in ("spec", "recompress"):
            for u in targets:
                self.anchor(u, inv)
            done = await asyncio.gather(*(self.spec_repair(u, usage, verify=cfg["repair"] == "spec", date=date)
                                          for u in targets))
            return {"revised": sorted(revised), "repaired": [u["id"] for u, d in zip(targets, done) if d[0]],
                    "closure": [u["id"] for u in targets],
                    "accepted": sum(d[1] for d in done), "rejected": sum(d[2] for d in done)}
        done = await asyncio.gather(*(self.repair_unit(u, inv, date, mode, usage, with_facts=cfg["facts"],
                                                       span_local=cfg["span"]) for u in targets))
        for u, d in zip(targets, done):
            if d and mode in ("clf", "ncf", "self"):
                self.unit_lineage(u, mode, getattr(self, "clf", None))
        return {"revised": sorted(revised), "repaired": [u["id"] for u, d in zip(targets, done) if d],
                "closure": [u["id"] for u in targets]}

    # ------------------------------------------- correction-conditioned field repair
    async def _ccf_repair(self, revised, inv, revmap, date, cfg, usage):
        """Exact anchored closure; inside each unit the correction-conditioned field Phi(c -> c', u) gives the
        sentences this specific correction changes; only those are re-derived (span-local), the rest copied."""
        from ncf import claim_input
        targets = [u for u in self.units.values() if set(u["inputs"]) & revised]
        queries = []
        for u in targets:
            for o in revised & set(u["inputs"]):
                for n in inv[o]:
                    queries.append((u, o, claim_input(o, self.claims[o]["text"], self.claims[n]["text"])))
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        sel = []
        for u in targets:
            qs = [(o, q) for uu, o, q in queries if uu is u]
            F = await asyncio.to_thread(self.clf.field, u["text_cited"], [q for _, q in qs], u["level"])
            sup = {o: [int(k) for k in np.nonzero(row >= 0.5)[0]] for (o, _), row in zip(qs, F)}
            sup = {o: ks for o, ks in sup.items() if ks}
            if sup:
                u["support"] = sup
                u["lineage"]["ccf"] = list(u["inputs"])
                sel.append(u)
        done = await asyncio.gather(*(self.repair_unit(u, inv, date, "ccf", usage, with_facts=cfg["facts"],
                                                       span_local=True) for u in sel))
        return {"revised": sorted(revised), "repaired": [u["id"] for u, d in zip(sel, done) if d],
                "closure": [u["id"] for u in targets], "selected": [u["id"] for u in sel],
                "support": {u["id"]: u.get("support", {}) for u in sel}}

    # -------------------------------------------------------- neural commutator
    async def _comm_repair(self, revised, inv, revmap, date, cfg, usage):
        if cfg.get("verify"):
            return await self._comm_verified_repair(revised, inv, revmap, date, cfg, usage)
        if cfg.get("fallback"):
            return await self._comm_ledger_repair(revised, inv, revmap, date, cfg, usage)
        return await self._comm_local_repair(revised, inv, revmap, date, cfg, usage)

    async def _comm_ledger_repair(self, revised, inv, revmap, date, cfg, usage):
        """Sufficiency-aware commutator.  T(delta, s) returns a copy, a corrected sentence, or LEDGER (the sentence
        does not determine its own repair).  LEDGER sentences of a unit are re-derived in one call of the
        compressor that reads the unit's anchored claims (ledger fallback); all others are local."""
        from commutator import T_PROMPT3, LEDGER, norm as tnorm
        targets = [u for u in self.units.values() if set(u["inputs"]) & revised]
        deltas = {o: [(o, self.claims[o]["text"], n, self.claims[n]["text"]) for n in inv[o]] for o in revised}
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        tu = Usage()
        rewritten, fell = {}, {}

        async def fix(u):
            text = u["text_cited"]
            sents = [text[a:b].strip() for a, b in split_sentences(text)]
            n0 = nl = 0
            for o in [c for c in u["inputs"] if c in revised]:
                for oid, old, nid, new in deltas[o]:
                    outs = await asyncio.gather(*(self.tnet.chat(T_PROMPT3.format(old=f"[{oid}] {old}", new=new, sent=s),
                                                                 max_tokens=min(int(len(s) / 3.2) + 64, 1000), usage=tu)
                                                  for s in sents))
                    outs = [x.strip().split("\n")[0].strip().strip('"') for x in outs]
                    led = [k for k, x in enumerate(outs) if x.startswith(LEDGER)]
                    if led and cfg["fallback"] == "copy":      # ablation: no ledger read, keep the sentence
                        for k in led:
                            outs[k] = sents[k]
                        led = []
                    if led:
                        nl += len(led)
                        anchored = [nid if c == oid else c for c in u["inputs"]]
                        got = await ledger_call(self.llm, u["level"], f"[{oid}] {old}", f"[{nid}] {new}",
                                                self._fact_lines([c for c in anchored if c in self.claims]), sents, led, usage)
                        for k in led:
                            outs[k] = got.get(k, sents[k])
                    for k, (s, x) in enumerate(zip(sents, outs)):
                        if x and tnorm(x) != tnorm(s):
                            sents[k] = x
                            n0 += 1
            if n0:
                u["text_cited"] = " ".join(s for s in sents if s)
                u["text"], u["revised_at"] = strip_cites(u["text_cited"]), date
                rewritten[u["id"]] = n0
            if nl:
                fell[u["id"]] = nl
            self.anchor(u, inv)

        await asyncio.gather(*(fix(u) for u in targets))
        return {"revised": sorted(revised), "repaired": sorted(rewritten), "closure": [u["id"] for u in targets],
                "rewritten_sents": rewritten, "ledger_sents": fell, "t_usage": tu.as_dict(),
                "closure_sents": sum(len(split_sentences(u["text_cited"])) for u in targets)}

    def _derivation_scope(self, u, cids, sents, ks, full=False):
        """Claims a non-local sentence of u may need: the inputs of the finest anchored unit that contains the
        corrected claim (its note or topic digest: the scope in which totals, maxima and orders are computed),
        plus the claims the sentences cite.  Bounded by a topic cluster, not by the history."""
        fine = [v for v in self.units.values() if v["level"] != "profile" and set(v["inputs"]) & cids]
        fine = sorted(fine, key=lambda v: (v["level"] != "digest", len(v["inputs"])))
        scope = list(u["inputs"]) if full or u["level"] != "profile" or not fine else list(fine[0]["inputs"])
        for k in ks:
            scope += [c for c in cited_ids(sents[k]) if c in self.claims]
        return list(dict.fromkeys(c for c in scope if c in self.claims))

    async def _comm_verified_repair(self, revised, inv, revmap, date, cfg, usage):
        """CoCoMem (v7).  Every closure sentence gets the local commutator's proposal T(delta, s) (tnet2) and the
        sufficiency verdict of the sufficiency-aware commutator (tnet3), in one parallel batch.  Sentences declared
        insufficient are verified by the compressor against their derivation scope, starting from the local
        proposal (a draft), so the fallback can only add the information the sentence lacked."""
        from commutator import T_PROMPT, T_PROMPT3, LEDGER, norm as tnorm
        global _SUFF
        single = bool(cfg.get("single"))
        if _SUFF is None and not single:
            from llm import LLM
            _SUFF = LLM(cfg["suff_name"], int(__import__("os").environ.get("CM_PS", cfg.get("suff_port", 18433))), concurrency=256)
        targets = [u for u in self.units.values() if set(u["inputs"]) & revised]
        deltas = {o: [(o, self.claims[o]["text"], n, self.claims[n]["text"]) for n in inv[o]] for o in revised}
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        tu = Usage()
        rewritten, fell = {}, {}
        mt = lambda s: min(int(len(s) / 3.2) + 64, 1000)

        async def fix(u):
            text = u["text_cited"]
            sents = [text[a:b].strip() for a, b in split_sentences(text)]
            n0 = nl = 0
            for o in [c for c in u["inputs"] if c in revised]:
                for oid, old, nid, new in deltas[o]:
                    if single:       # one network, three outcomes: copy / corrected sentence / [LEDGER] <local draft>
                        outs = await asyncio.gather(*(self.tnet.chat(T_PROMPT3.format(old=f"[{oid}] {old}", new=new, sent=s),
                                                                     max_tokens=mt(s) + 16, usage=tu) for s in sents))
                        outs = [x.strip().split("\n")[0].strip().strip('"') for x in outs]
                        led = [k for k, x in enumerate(outs) if x.startswith(LEDGER)]
                        draft = [((x[len(LEDGER):].strip().strip('"') if x.startswith(LEDGER) else x) or s)
                                 for x, s in zip(outs, sents)]
                    else:
                        loc, suf = await asyncio.gather(
                            asyncio.gather(*(self.tnet.chat(T_PROMPT.format(old=f"[{oid}] {old}", new=new, sent=s),
                                                            max_tokens=mt(s), usage=tu) for s in sents)),
                            asyncio.gather(*(_SUFF.chat(T_PROMPT3.format(old=f"[{oid}] {old}", new=new, sent=s),
                                                        max_tokens=8, usage=tu) for s in sents)))
                        draft = [(x.strip().split("\n")[0].strip().strip('"') or s) for x, s in zip(loc, sents)]
                        led = [k for k, x in enumerate(suf) if x.strip().startswith(LEDGER)]
                    if led and cfg["verify"] != "local":
                        nl += len(led)
                        scope = self._derivation_scope(u, {o}, sents, led, full=cfg.get("scope") == "full")
                        scope = [nid if c == oid else c for c in scope]
                        got = await ledger_call(self.llm, u["level"], f"[{oid}] {old}", f"[{nid}] {new}",
                                                self._fact_lines([c for c in dict.fromkeys(scope) if c in self.claims]),
                                                draft, led, usage, think=bool(cfg.get("think_fb")))
                        for k in led:
                            if got.get(k):
                                draft[k] = got[k]
                    for k, (s, x) in enumerate(zip(sents, draft)):
                        if x and tnorm(x) != tnorm(s):
                            sents[k] = x
                            n0 += 1
            if n0:
                u["text_cited"] = " ".join(s for s in sents if s)
                u["text"], u["revised_at"] = strip_cites(u["text_cited"]), date
                rewritten[u["id"]] = n0
            if nl:
                fell[u["id"]] = nl
            self.anchor(u, inv)

        await asyncio.gather(*(fix(u) for u in targets))
        return {"revised": sorted(revised), "repaired": sorted(rewritten), "closure": [u["id"] for u in targets],
                "rewritten_sents": rewritten, "ledger_sents": fell, "t_usage": tu.as_dict(),
                "closure_sents": sum(len(split_sentences(u["text_cited"])) for u in targets)}

    async def _comm_local_repair(self, revised, inv, revmap, date, cfg, usage):
        """Exact anchored closure; every sentence of every affected unit is passed through the commutator
        T(delta, s) -> s' (one Jacobi step of the compressor, distilled).  Copy = independent of delta.
        cfg["tnet"] == "student": the distilled network (self.tnet); "teacher": the compressor prompted."""
        from commutator import T_PROMPT, norm as tnorm
        targets = [u for u in self.units.values() if set(u["inputs"]) & revised]
        deltas = {o: [(f"[{o}] {self.claims[o]['text']}", self.claims[n]["text"]) for n in inv[o]] for o in revised}
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        student = cfg["tnet"] == "student"
        T, tu = (self.tnet, Usage()) if student else (self.llm, usage)
        rewritten = {}

        async def fix(u):
            text = u["text_cited"]
            sents = [text[a:b].strip() for a, b in split_sentences(text)]
            n0 = 0
            for o in [c for c in u["inputs"] if c in revised]:
                for old, new in deltas[o]:
                    outs = await asyncio.gather(*(T.chat(T_PROMPT.format(old=old, new=new, sent=s),
                                                         max_tokens=min(int(len(s) / 3.2) + 64, 1000), usage=tu) for s in sents))
                    for k, (s, x) in enumerate(zip(sents, outs)):
                        x = x.strip().split("\n")[0].strip().strip('"')
                        if x and tnorm(x) != tnorm(s):
                            sents[k] = x
                            n0 += 1
            if n0:
                u["text_cited"] = " ".join(s for s in sents if s)
                u["text"], u["revised_at"] = strip_cites(u["text_cited"]), date
                rewritten[u["id"]] = n0
            self.anchor(u, inv)

        await asyncio.gather(*(fix(u) for u in targets))
        info = {"revised": sorted(revised), "repaired": sorted(rewritten), "closure": [u["id"] for u in targets],
                "rewritten_sents": rewritten,
                "closure_sents": sum(len(split_sentences(u["text_cited"])) for u in targets)}
        if student:
            info["t_usage"] = tu.as_dict()
        return info

    # ------------------------------------------- v20: witness-conditioned commutator
    def _witness_lines(self, ids, exclude):
        """Dereference a witness against the CURRENT ledger: superseded claims are read through their successors, so the
        operands are current also after earlier corrections (sequential maintenance needs no witness rewriting)."""
        out, seen = [], set()
        pending = list(reversed(ids))
        while pending:
            c = pending.pop()
            if c in seen or c not in self.claims:
                continue
            seen.add(c)
            x = self.claims[c]
            if x["status"] == "superseded":
                pending.extend(reversed(x.get("superseded_by", [])))
                continue
            if c in exclude:
                continue
            out.append(f"[{c}] ({x['date'][:10]}, {x['speaker']}) {x['text']}")
        return list(dict.fromkeys(out))

    async def _ensure_witness(self, u, usage):
        """Units written after the write-time witness pass (none in our runs) are annotated on demand; the tokens are
        reported separately as write cost."""
        from witness import annotate_unit
        u["sents"], u["witness"] = await annotate_unit(self.llm, self, u, usage)

    async def _witness_repair(self, revised, inv, revmap, date, cfg, usage):
        """CoCoMem (v20).  Exact anchored closure A(c); every sentence s_k of every unit in A(c) is mapped, in ONE parallel
        round, by the witness-conditioned commutator T(delta, s_k, W_k), where W_k is the sentence's derivation witness
        dereferenced in the current ledger.  A copy keeps the sentence bit-for-bit.  The witness is anchored with the unit:
        the successor claim takes the slot of the corrected one.  No ledger read of the unit, no fallback, no trigger.
        cfg["witness"]: "annot" (write-time witness), "cite" (only the claims the sentence cites; ablation),
        "none" (two-input commutator; ablation)."""
        from commutator import W_PROMPT, T_PROMPT, render_wfacts, first_line, out_budget, norm as tnorm
        mode = cfg.get("witness", "annot")
        targets = [u for u in self.units.values() if set(u["inputs"]) & revised]
        deltas = {o: [(o, self.claims[o]["text"], n, self.claims[n]["text"]) for n in inv[o]] for o in revised}
        wu = Usage()
        if mode == "annot":
            await asyncio.gather(*(self._ensure_witness(u, wu) for u in targets if "witness" not in u))
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        student = cfg["tnet"] == "student"
        T, tu = (self.tnet, Usage()) if student else (self.llm, usage)
        rewritten, cut = {}, 0

        async def fix(u):
            nonlocal cut
            sents = list(u.get("sents") or [u["text_cited"][a:b].strip() for a, b in split_sentences(u["text_cited"])])
            W = [list(w) for w in (u.get("witness") or [])]
            W = (W + [[] for _ in sents])[:len(sents)]
            n0 = 0
            for o in [c for c in u["inputs"] if c in revised]:
                for oid, old, nid, new in deltas[o]:
                    def prompt(k, s):
                        if mode == "none":
                            return T_PROMPT.format(old=f"[{oid}] {old}", new=new, sent=s)
                        ids = W[k] if mode == "annot" else cited_ids(s)
                        lines = self._witness_lines(ids, exclude={oid, nid})
                        facts = render_wfacts(lines)
                        return W_PROMPT.format(old=f"[{oid}] {old}", new=new, facts=facts, sent=s), len(facts) < sum(map(len, lines))
                    ps = [prompt(k, s) for k, s in enumerate(sents)]
                    if mode != "none":
                        cut += sum(c for _, c in ps)
                        ps = [p for p, _ in ps]
                    outs = await asyncio.gather(*(T.chat(p, max_tokens=out_budget(s), usage=tu) for p, s in zip(ps, sents)))
                    for k, (s, x) in enumerate(zip(sents, outs)):
                        x = first_line(x)
                        if x and tnorm(x) != tnorm(s):
                            sents[k] = x
                            n0 += 1
            # anchoring of the witness: successors take the slots of the claims they revise
            W = [list(dict.fromkeys(n for c in w for n in (inv.get(c) or [c]))) for w in W]
            u["sents"], u["witness"] = sents, W
            if n0:
                u["text_cited"] = " ".join(s for s in sents if s)
                u["text"], u["revised_at"] = strip_cites(u["text_cited"]), date
                rewritten[u["id"]] = n0
            self.anchor(u, inv)

        await asyncio.gather(*(fix(u) for u in targets))
        info = {"revised": sorted(revised), "repaired": sorted(rewritten), "closure": [u["id"] for u in targets],
                "rewritten_sents": rewritten, "witness_cut": cut,
                "closure_sents": sum(len(u.get("sents") or []) for u in targets),
                "witness_write_usage": wu.as_dict()}
        if student:
            info["t_usage"] = tu.as_dict()
        return info

    # ------------------------------------------- v21: materialized derivation records
    async def _ensure_deriv(self, u, usage):
        """Units written after the write-time derivation pass (none in our runs) are annotated on demand; the tokens
        are reported separately as write cost."""
        from deriv import annotate_unit
        from llm import ContextOverflow
        try:
            u["sents"], u["deriv"], u["dids"] = await annotate_unit(self.llm, self, u, usage)
        except ContextOverflow:
            # the compressor serving this run has a reduced context window (shared GPU); the unit keeps empty
            # records and the fact is counted in the run's info (precomputed records make this a no-op)
            sents = [u["text_cited"][a:b].strip() for a, b in split_sentences_v2(u["text_cited"])]
            u["sents"], u["deriv"], u["dids"], u["deriv_missing"] = sents, ["none"] * len(sents), [[] for _ in sents], True

    async def _deriv_repair(self, revised, inv, revmap, date, cfg, usage):
        """CoCoMem (v21).  Every stored sentence s_k carries a MATERIALIZED derivation record D_k, written by the
        compressor at write time: the claims the sentence was derived from, each with the value the sentence took
        from it, and the computation (total, count, extreme, order, duration, conclusion).  D_k is the auxiliary view
        that makes the sentence self-maintainable in the classical sense: the repair reads NOTHING but the correction,
        the sentence and its record -- no ledger access, no dereferencing.  One network maps (delta, D_k, s_k) to
        (D'_k, s'_k) in one parallel round over the exact anchored closure A(c): it rewrites the record first (the
        delta rule made explicit: the corrected operand replaces the old one and the computation is redone) and then
        re-states the sentence.  Copies keep record and sentence bit for bit.  Records are anchored like the inputs:
        the successor claim takes the identifier of the claim it revises.
        cfg["drec"]: "rec" (record in the input, record+sentence in the output; CoCoMem), "ptr" (ablation: the record's
        claim ids dereferenced in the ledger, sentence output only -- the pointer witness of v20), "none" (two-input
        commutator).  cfg["select"]: "all" (every closure sentence is mapped) or "rec" (only sentences whose record or
        citations name the corrected claim: the write-time sentence closure S(c) of Theorem 2)."""
        from commutator21 import D_PROMPT, parse_two, record_ids, anchor_record, out_budget_d, upsert_record, foreign_entries
        from commutator import T_PROMPT, W_PROMPT, render_wfacts, first_line, out_budget, norm as tnorm
        mode, select = cfg.get("drec", "rec"), cfg.get("select", "all")
        keyed = cfg.get("keyed", False)          # the record is keyed by claim: a correction of c upserts the entry of c
        rec_writes = collections.Counter()
        targets = [u for u in self.units.values() if set(u["inputs"]) & revised]
        deltas = {o: [(o, self.claims[o]["text"], n, self.claims[n]["text"]) for n in inv[o]] for o in revised}
        wu = Usage()
        if mode != "none":                       # the two-input ablation reads no record
            await asyncio.gather(*(self._ensure_deriv(u, wu) for u in targets if "deriv" not in u))
        self.apply_versions(revmap, date, keep_history=cfg["history"])
        student = cfg["tnet"] == "student"
        T, tu = (self.tnet, Usage()) if student else (self.llm, usage)
        rewritten, called, total = {}, 0, 0
        missing = sum(bool(u.get("deriv_missing")) for u in targets)

        async def fix(u):
            nonlocal called, total
            sents = list(u.get("sents") or [u["text_cited"][a:b].strip() for a, b in split_sentences_v2(u["text_cited"])])
            D = (list(u.get("deriv") or []) + ["none"] * len(sents))[:len(sents)]
            ids = [record_ids(d) | set(cited_ids(s)) for d, s in zip(D, sents)]
            n0 = 0
            for o in [c for c in u["inputs"] if c in revised]:
                for oid, old, nid, new in deltas[o]:
                    ks = [k for k in range(len(sents)) if select == "all" or oid in ids[k]]
                    total += len(sents)

                    async def call(k):
                        s = sents[k]
                        corr = f"[{oid}] {old}"
                        if mode == "none":
                            x = first_line(await T.chat(T_PROMPT.format(old=corr, new=new, sent=s), max_tokens=out_budget(s), usage=tu))
                            return k, x, None
                        if mode == "ptr":
                            lines = self._witness_lines(sorted(ids[k], key=lambda c: int(c[1:])), exclude={oid, nid})
                            x = first_line(await T.chat(W_PROMPT.format(old=corr, new=new, facts=render_wfacts(lines), sent=s),
                                                        max_tokens=out_budget(s), usage=tu))
                            return k, x, None
                        r = await T.chat(D_PROMPT.format(old=corr, new=new, rec=D[k], sent=s), max_tokens=out_budget_d(s, D[k]), usage=tu)
                        d2, x = parse_two(r, D[k], s)
                        return k, x, d2

                    outs = await asyncio.gather(*(call(k) for k in ks))
                    called += len(ks)
                    for k, x, d2 in outs:
                        if x and tnorm(x) != tnorm(sents[k]):
                            sents[k] = x
                            n0 += 1
                        if d2:
                            if tnorm(d2) != tnorm(D[k]):
                                rec_writes["records"] += 1
                                rec_writes["other_operand_altered"] += foreign_entries(d2, oid) != foreign_entries(D[k], oid)
                            D[k] = upsert_record(D[k], d2, oid) if keyed else d2
                    D = [anchor_record(d, oid, nid) for d in D]
                    ids = [{nid if c == oid else c for c in S} for S in ids]
            u["sents"], u["deriv"], u["dids"] = sents, D, [sorted(S, key=lambda c: int(c[1:])) for S in ids]
            if n0:
                u["text_cited"] = " ".join(s for s in sents if s)
                u["text"], u["revised_at"] = strip_cites(u["text_cited"]), date
                rewritten[u["id"]] = n0
            self.anchor(u, inv)

        await asyncio.gather(*(fix(u) for u in targets))
        info = {"revised": sorted(revised), "repaired": sorted(rewritten), "closure": [u["id"] for u in targets],
                "rewritten_sents": rewritten, "closure_sents": total, "mapped_sents": called,
                "deriv_write_usage": wu.as_dict(), "deriv_missing_units": missing,
                "rec_writes": rec_writes["records"], "rec_other_operand_altered": rec_writes["other_operand_altered"]}
        if student:
            info["t_usage"] = tu.as_dict()
        return info

    # ------------------------------------------------- speculative recompression
    def anchor(self, u, inv):
        """Anchoring: each corrected claim's successor takes its slot in the unit's input list."""
        out = []
        for c in u["inputs"]:
            for n in (inv.get(c) or [c]):
                if n not in out:
                    out.append(n)
        u["inputs"] = out

    def unit_prompt(self, u):
        """The exact compression prompt of unit u over its (current) anchored inputs (in the configured mode)."""
        return self._unit_prompt_for(u["level"], [c for c in u["inputs"] if c in self.claims], date=u.get("date"),
                                     person=u.get("person"))

    def _set_compressed(self, u, w, date=None):
        """Replace the text-dependent fields of unit u by a fresh compression result (recompress / rebuild)."""
        for k in ("sents", "deriv", "dids", "plan_ok"):
            u.pop(k, None)
        u.update(w)
        if date is not None:
            u["revised_at"] = date

    async def spec_repair(self, u, usage, verify=True, date=None):
        """Correction as warm-started Jacobi decoding of the compressor.

        Greedy compression of the corrected history is the fixed point of u_k = f_k(I'; u_<k).  The stale
        unit is a near-fixed point (a draft); one parallel Jacobi step re-derives every sentence with the
        compressor under the corrected, anchored inputs I', teacher-forced on the draft prefix, and keeps
        the draft sentence bit-for-bit whenever the re-derived one states the same information (the
        speculative-decoding acceptance test, at sentence granularity).  verify=False recompresses the unit
        from its corrected inputs (local recompression).  Returns (changed, accepted, rejected)."""
        prompt = self.unit_prompt(u)
        mt = {"note": 400, "digest": 600, "profile": 900}[u["level"]]
        if not verify:                       # local recompression C(delta(H)) on the closure (records too, in plan mode)
            w = await self._compress(u["level"], [c for c in u["inputs"] if c in self.claims], usage, date=u.get("date"),
                                     person=u.get("person"))
            self._set_compressed(u, w, date)
            return (True, 0, 0)
        if COMPRESSOR == "plan":
            raise NotImplementedError("Jacobi re-derivation (spec) is defined on the text grammar only")
        old = u["text_cited"]
        rng = split_sentences(old)
        if not rng:
            return (False, 0, 0)
        sents = [old[a:b] for a, b in rng]
        seps = [old[rng[i][1]:rng[i + 1][0]] for i in range(len(rng) - 1)] + [""]
        lead = old[:rng[0][0]]
        pre = chat_prefix(prompt)
        npre = ntok_exact(pre)
        # one Jacobi step from the stale draft: every sentence is re-derived in parallel by the compressor
        # under the corrected inputs, teacher-forced on the draft prefix
        prefixes = [lead + "".join(sents[i] + seps[i] for i in range(j)) for j in range(len(sents))]
        raw = await asyncio.gather(*(self.llm.complete(pre + prefixes[j], max_tokens=len(sents[j]) // 3 + 24)
                                     for j in range(len(sents))))
        # cost with KV reuse (one shared prefill of prompt + draft, as in speculative/Jacobi decoding)
        usage.add(npre + ntok_exact(old), sum(ntok_exact(r) for r in raw))
        cands = [first_sentence(r) for r in raw]
        same = await asyncio.gather(*(self._same(x, c, usage) for x, c in zip(sents, cands)))
        out = [x if ok else c for x, c, ok in zip(sents, cands, same)]
        acc, rej = sum(same), len(same) - sum(same)
        if rej == 0:
            return (False, acc, 0)
        txt = (lead + "".join(o + sp for o, sp in zip(out, seps) if o.strip())).strip()
        u["text_cited"], u["text"], u["revised_at"] = txt, strip_cites(txt), date
        return (True, acc, rej)

    async def _same(self, a, b, usage):
        def n(x):
            return re.sub(r"[^a-z0-9]", "", strip_cites(x).lower())
        if n(a) == n(b):
            return True
        if not b.strip():
            return False
        r = await self.llm.chat(prompts.SAME.format(a=strip_cites(a), b=strip_cites(b)), max_tokens=4, usage=usage)
        return "no" not in r.lower()

    async def _rebuild(self, usage):
        """Reference: recompress every derived unit from the corrected ledger (C(delta(H)))."""
        notes = [u for u in self.units.values() if u["level"] == "note"]
        for u in notes:
            u["inputs"] = [c for c in self._current_inputs(u["inputs"])]

        async def renote(u):
            if not u["inputs"]:
                return
            self._set_compressed(u, await self._compress("note", u["inputs"], usage, date=u["date"]))
        await asyncio.gather(*(renote(u) for u in notes))
        for k in [k for k, u in self.units.items() if u["level"] in ("digest", "profile")]:
            del self.units[k]
        await self.consolidate(usage)

    def _current_inputs(self, cids):
        out = []
        for c in cids:
            x = self.claims.get(c)
            if x is None:
                continue
            if x["status"] == "superseded":
                out += [n for n in x["superseded_by"] if n not in out]
            elif c not in out:
                out.append(c)
        return out

    async def _correct_unit_level(self, new, date, strategy, usage, k=5):
        """Baselines operating on unit text without lineage: Mem0-style similarity update, or a full LLM scan."""
        items = [("claim", c, self.claims[c]["text"]) for c in self.claims if c not in new and self.claims[c]["status"] == "current"]
        items += [("unit", u["id"], u["text"]) for u in self.units.values()]
        newtxt = "\n".join(f"- {self.claims[c]['text']}" for c in new)
        if not new:
            return {"revised": [], "repaired": []}
        if strategy == "simupdate":
            P = await emb([t for _, _, t in items])
            Q = await emb([self.claims[c]["text"] for c in new], query=True)
            cand = set()
            for i in range(len(new)):
                for j in np.argsort(-(Q[i] @ P.T))[:k]:
                    cand.add(j)
            targets = [items[j] for j in sorted(cand)]
        else:
            targets = items
        repaired = []

        async def one(kind, iid, text):
            r = await self.llm.chat(UPDATE_ITEM.format(item=text, new=newtxt, date=date), max_tokens=500, usage=usage)
            js = parse_json(r, {})
            if not isinstance(js, dict):
                return
            op = str(js.get("op", "NONE")).upper()
            if op == "UPDATE" and js.get("text"):
                if kind == "claim":
                    self.claims[iid]["text"] = js["text"]
                else:
                    self.units[iid]["text"] = strip_cites(js["text"])
                    self.units[iid]["text_cited"] = js["text"]
                repaired.append(iid)
            elif op == "DELETE":
                if kind == "claim":
                    self.claims.pop(iid, None)
                else:
                    self.units.pop(iid, None)
                repaired.append(iid)

        await asyncio.gather(*(one(*t) for t in targets))
        return {"revised": [], "repaired": repaired}

    # ------------------------------------------------------------------ read
    def slot_text(self, c):
        """A claim slot as read: its current value plus its version chain (anchoring at read time)."""
        x = self.claims[c]
        chain, seen = [], set()
        stack = list(x.get("revises", []))
        while stack:
            o = stack.pop(0)
            if o in seen or o not in self.claims:
                continue
            seen.add(o)
            chain.append(self.claims[o])
            stack += self.claims[o].get("revises", [])
        if not chain:
            return x["text"]
        prev = "; ".join(f'previously recorded on {y["date"][:10]}: "{y["text"]}"' for y in chain)
        return f'{x["text"]} [corrected on {x["date"][:10]}; {prev}]'

    async def retrieve(self, question, budget=4000, include_superseded=True, levels=None):
        """levels: restrict the context to these levels, e.g. ("note", "digest") reads the derived views alone."""
        items = []
        for c, x in self.claims.items():
            if x["status"] == "superseded":
                # folded into the slot of its successor (when history is kept, the successor carries it)
                if not include_superseded or any(n in self.claims for n in x.get("superseded_by", [])):
                    continue
            # a slot is listed under the date of its first record (the correction date is part of the slot text)
            items.append(("fact", x.get("since") or x["date"], self.slot_text(c), x["status"] == "superseded", x.get("superseded_at")))
        for u in self.units.values():
            if u["level"] == "profile":
                continue
            # a maintained unit reads like the unit written from the corrected history, which carries no revision mark
            items.append((u["level"], u["date"], u["text"], False, u.get("revised_at") if REVISED_TAG else None))
        if levels is not None:
            items = [t for t in items if t[0] in levels]
        if not items:
            return [], ""
        E = await emb([t[2] for t in items])
        q = (await emb([question], query=True))[0]
        order = np.argsort(-(E @ q))
        profs = [u for u in self.units.values() if u["level"] == "profile" and (levels is None or "profile" in levels)]
        used = sum(ntok(p["text"]) for p in profs)
        chosen = []
        for i in order:
            t = ntok(items[i][2]) + 12
            if used + t > budget:
                continue
            chosen.append(items[i])
            used += t
        chosen.sort(key=lambda z: z[1])
        lines = []
        for prof in profs:
            rv = f" (revised {prof['revised_at']})" if prof.get("revised_at") else ""
            who = f"Profile of {prof['person']}" if prof.get("person") else "User profile"
            lines.append(f"[{who}{rv}] {prof['text']}")
        for lev, date, text, sup, when in chosen:
            if sup:
                lines.append(f"[{date}] ({lev}) [SUPERSEDED on {when}] {text}")
            elif when:
                lines.append(f"[{date}] ({lev}, revised {when}) {text}")
            else:
                lines.append(f"[{date}] ({lev}) {text}")
        return chosen, "\n".join(lines)

    async def answer(self, question, qdate, budget=4000):
        _, ctx = await self.retrieve(question, budget)
        return await reader_answer(self.llm, ctx, question, qdate, self.usage["read"], persons=self.persons)

    # --------------------------------------------------------------- serialize
    def to_json(self):
        return {"claims": self.claims, "units": self.units, "n": self.n, "persons": self.persons,
                "compressor": COMPRESSOR, "usage": {k: v.as_dict() for k, v in self.usage.items()}}

    @classmethod
    def from_json(cls, llm, d, **kw):
        m = cls(llm, persons=d.get("persons"), **kw)
        m.claims = copy.deepcopy(d["claims"])
        m.units = copy.deepcopy(d["units"])
        m.n = d["n"]
        return m

    def clone(self):
        m = CoCoMem(self.llm, self.note_words, self.digest_words, self.profile_words, self.cluster_dist, self.persons)
        m.claims, m.units, m.n = copy.deepcopy(self.claims), copy.deepcopy(self.units), self.n
        m.text_words = dict(self.text_words)
        m.clf = getattr(self, "clf", None)
        m.tnet = getattr(self, "tnet", None)
        return m

    def size_tokens(self):
        """Stored tokens: the ledger, the unit texts, and the derivation records (the auxiliary view kept with the
        units; zero for memories without records)."""
        return {"claims": sum(ntok(x["text"]) for x in self.claims.values()),
                "units": sum(ntok(u["text"]) for u in self.units.values()),
                "records": sum(ntok(r) for u in self.units.values() for r in (u.get("deriv") or [])
                               if r and r.strip().lower() != "none")}


_SUFF = None


async def ledger_call(llm, level, old, new, facts, sents, ids, usage=None, think=False):
    """Ledger fallback: the compressor re-derives sentences `ids` of a unit from the unit's anchored claims.
    Returns {k: sentence}.  Also used to build the sufficiency labels of the commutator (commutator.py)."""
    shown = "\n".join(f"({k}) {s}" for k, s in enumerate(sents))
    r = await llm.chat(prompts.LEDGER_REPAIR.format(kind=level, old=old, new=new, facts=facts, unit=shown,
                                                    ids=", ".join(str(k) for k in ids)),
                       max_tokens=min(120 + sum(len(sents[k]) // 3 + 40 for k in ids), 3000) + (5000 if think else 0),
                       usage=usage, think=think)
    out = {}
    for line in r.splitlines():
        m = re.match(r"\s*\(?(\d+)\)?\s*[:.)]\s*(.*)$", line)
        if m and int(m.group(1)) in ids:
            out[int(m.group(1))] = m.group(2).strip().strip('"')
    return out


def _d21(name, **kw):
    base = {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "deriv", "tnet": "student",
            "tnet_name": name, "drec": "rec", "select": "all", "deriv": True}
    base.update(kw)
    return base


STRATEGIES = {
    # v22 (CoCoMem): record-first compression (every sentence is written from its derivation record, CM_COMPRESSOR=plan,
    # rebuild_units.py) + the same derivation-conditioned commutator, distilled on the record-first memories (tnet22).
    # The correction path is that of v21: exact anchored closure, one parallel round, no ledger access.  The record is
    # keyed by claim (commutator21.upsert_record): a correction of c writes the entry of c and the computation, and
    # carries the entries of the other operands over bit for bit.
    "comm22": _d21("tnet22", keyed=True),
    "comm22_sel": _d21("tnet22", select="rec", keyed=True),           # only the write-time sentence closure S(c)
    "comm22_raw": _d21("tnet22"),                                     # - keyed record: the operator's whole record is stored
    "comm22_nd": _d21("tnet22nd", drec="none"),                       # - record (two-input student, same targets)
    "comm22_w": _d21("tnet22w", drec="ptr"),                          # record := pointers dereferenced in the ledger
    "comm22_nv": _d21("tnet22nv", keyed=True),                        # - commutation verification
    "comm22_nr": _d21("tnet22nr", keyed=True),                        # - retraction pairs
    "comm22_overwrite": _d21("tnet22", history=False, keyed=True),    # - version slots
    "comm22_zs": _d21(None, tnet="teacher", keyed=True),              # 8B compressor zero-shot as the same operator
    "comm22_t21": _d21("tnet21", keyed=True),                         # the v21 student on the record-first memories
    "comm22_s1": _d21("tnet22_s1", keyed=True),
    "comm22_s2": _d21("tnet22_s2", keyed=True),
    "comm22_blind": _d21("tnet22", intake="ground", keyed=True),      # - ledger-conditioned intake (claims from the session alone)
    # v21 (CoCoMem): exact anchored closure + materialized derivation records + derivation-conditioned commutator
    # T(delta, D_k, s_k) -> (D'_k, s'_k) (tnet21); one parallel round; no ledger access at correction time
    "comm21": _d21("tnet21"),
    "comm21np": _d21("tnet21", keyed=True),                           # the v22 correction path on the v21 memories without the profile
    "comm21np_sel": _d21("tnet21", select="rec", keyed=True),         # the same, only the sentences whose attributed record names c
    "comm21_sel": _d21("tnet21", select="rec"),                       # only the write-time sentence closure S(c)
    "comm21_nd": _d21("tnet21nd", drec="none"),                       # - record (two-input student, same targets)
    "comm21_w": _d21("tnet21w", drec="ptr"),                          # record := pointers dereferenced in the ledger (v20)
    "comm21_nv": _d21("tnet21nv"),                                    # - commutation verification
    "comm21_nr": _d21("tnet21nr"),                                    # - retraction pairs
    "comm21_overwrite": _d21("tnet21", history=False),                # - version slots
    "comm21_zs": _d21(None, tnet="teacher"),                          # 8B compressor zero-shot as the same operator
    "comm21_s1": _d21("tnet21_s1"),                                   # training seeds 1 and 2 of the main student
    "comm21_s2": _d21("tnet21_s2"),
    # v20 (CoCoMem): exact anchored closure + witness-conditioned commutator T(delta, s, W) (tnet20), one parallel round
    "comm20": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "witness", "tnet": "student",
               "tnet_name": "tnet20", "witness": "annot"},
    # ablations: one component removed each
    "comm20_nw": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "witness",
                  "tnet": "student", "tnet_name": "tnet20nw", "witness": "none"},        # - witness (input and training)
    "comm20_cite": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "witness",
                    "tnet": "student", "tnet_name": "tnet20", "witness": "cite"},      # witness := inline citations
    "comm20_nv": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "witness",
                  "tnet": "student", "tnet_name": "tnet20nv", "witness": "annot"},       # - commutation verification
    "comm20_nr": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "witness",
                  "tnet": "student", "tnet_name": "tnet20nr", "witness": "annot"},       # - retraction pairs
    "comm20_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "witness",
                         "tnet": "student", "tnet_name": "tnet20", "witness": "annot"},  # - version slots
    "comm20_zs": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "witness",
                  "tnet": "teacher", "witness": "annot"},                                # 8B zero-shot, same operator
    # v7 (CoCoMem): local commutator (tnet2) + sufficiency verdict (tnet3) + compressor verification of the
    # insufficient sentences against their derivation scope, starting from the local draft
    # v8 (CoCoMem): student distilled from well-formed information-level targets ("tnet5"); comm6 adds the
    # sufficiency verdict and compressor verification of insufficient sentences
    "comm5": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet5"},
    "comm5_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                        "tnet": "student", "tnet_name": "tnet5"},
    "comm6": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet5", "suff_name": "tnet3", "verify": True},
    # v9 (CoCoMem): student distilled from the reasoning compressor's targets ("tnet7")
    "comm7": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet7"},
    "comm7_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                        "tnet": "student", "tnet_name": "tnet7"},
    "comm8": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet7", "suff_name": "tnet3", "verify": True},
    # v10 (CoCoMem): student distilled from commutation-verified targets ("tnet10", commutator.py relabel_cv)
    "comm10": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
               "tnet_name": "tnet10"},
    "comm10_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                         "tnet": "student", "tnet_name": "tnet10"},
    "comm11": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
               "tnet_name": "tnet10", "suff_name": "tnet3", "verify": True},
    # v12 (CoCoMem): ONE student with three outcomes ("tnet12"); the modes differ only in what happens at [LEDGER]
    "comm13": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
               "tnet_name": "tnet13", "single": True, "verify": "local"},
    "comm13L": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
                "tnet_name": "tnet13", "single": True, "verify": True},
    "comm13T": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
                "tnet_name": "tnet13", "single": True, "verify": True, "think_fb": True},
    "comm12": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
               "tnet_name": "tnet12", "single": True, "verify": "local"},
    "comm12_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                         "tnet": "student", "tnet_name": "tnet12", "single": True, "verify": "local"},
    "comm12L": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
                "tnet_name": "tnet12", "single": True, "verify": True},
    "comm12T": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
                "tnet_name": "tnet12", "single": True, "verify": True, "think_fb": True},
    "comm12F": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
                "tnet_name": "tnet12", "single": True, "verify": True, "think_fb": True, "scope": "full"},
    "comm4": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet2", "suff_name": "tnet3", "verify": True},
    "comm4_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                        "tnet": "student", "tnet_name": "tnet2", "suff_name": "tnet3", "verify": True},
    # CoCoMem v4: exact anchored closure + neural commutator over every closure sentence + version slots
    "comm": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student"},
    "comm_zs": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "teacher"},
    "comm_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                       "tnet": "student"},
    # v5: student distilled from information-level targets (served as "tnet2")
    "comm2": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet2"},
    "comm2_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                        "tnet": "student", "tnet_name": "tnet2"},
    # v6 (CoCoMem): sufficiency-aware commutator (copy / local repair / LEDGER) + ledger fallback ("tnet3")
    "comm3": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm", "tnet": "student",
              "tnet_name": "tnet3", "fallback": True},
    "comm3_nofallback": {"lineage": "coarse", "span": True, "facts": False, "history": True, "repair": "comm",
                         "tnet": "student", "tnet_name": "tnet3", "fallback": "copy"},
    "comm3_overwrite": {"lineage": "coarse", "span": True, "facts": False, "history": False, "repair": "comm",
                        "tnet": "student", "tnet_name": "tnet3", "fallback": True},
    # name: lineage source, span-local repair, local decompression (lineage facts), keep version history
    "ncf": {"lineage": "ncf", "span": True, "facts": True, "history": True},              # CoCoMem (full)
    "ccf": {"lineage": "coarse", "span": True, "facts": True, "history": True, "repair": "ccf"},   # CoCoMem v3
    "slots": {"lineage": "coarse", "span": False, "facts": True, "history": True, "repair": "none"},
    "ccf_overwrite": {"lineage": "coarse", "span": True, "facts": True, "history": False, "repair": "ccf"},
    "ncf_unit": {"lineage": "ncf", "span": False, "facts": True, "history": True},        # - span-local repair
    "ncf_nofacts": {"lineage": "ncf", "span": True, "facts": False, "history": True},     # - local decompression
    "ncf_overwrite": {"lineage": "ncf", "span": True, "facts": True, "history": False},   # - version history
    "self_span": {"lineage": "self", "span": True, "facts": True, "history": True},       # cited sentences as span
    # CoCoMem: exact anchored closure + speculative recompression (draft = stale unit, verifier = compressor)
    "spec": {"lineage": "coarse", "span": False, "facts": True, "history": True, "repair": "spec"},
    "recompress": {"lineage": "coarse", "span": False, "facts": True, "history": True, "repair": "recompress"},
    "spec_overwrite": {"lineage": "coarse", "span": False, "facts": True, "history": False, "repair": "spec"},
    "spec_ncf": {"lineage": "ncf", "span": False, "facts": True, "history": True, "repair": "spec"},
    "coarse": {"lineage": "coarse", "span": False, "facts": True, "history": True},       # construction lineage
    "self": {"lineage": "self", "span": False, "facts": True, "history": True},           # LLM-cited lineage
    "rebuild": {"lineage": "coarse", "span": False, "facts": True, "history": True},      # recompress all
}


async def reader_answer(llm, ctx, question, qdate, usage, think=False, persons=None):
    if persons:
        pr = prompts.READER_DIALOG.format(a=persons[0], b=persons[1], memory=ctx or "(empty)", question=question)
    else:
        pr = prompts.READER.format(memory=ctx or "(empty)", date=qdate, question=question)
    r = await llm.chat(pr, max_tokens=3000 if think else 400, usage=usage, think=think)
    m = re.search(r"Answer:\s*(.*)", r, flags=re.S)
    return (m.group(1).strip() if m else r.strip()), r


PROFILE = """Below are facts the user shared across many past conversations (chronological). Write the user's profile (at most {words} words): who they are, where they live and work, family, pets, possessions, ongoing projects, routines, preferences and the current state of anything that changed over time. Keep specific values (names, numbers, dates). After each sentence, cite in square brackets the ids of the facts it is based on.

Facts:
{facts}

User profile:"""

UPDATE_ITEM = """You are a smart memory manager. A new conversation (dated {date}) produced the following new information:
{new}

Existing memory entry:
\"\"\"{item}\"\"\"

Compare the new information with the existing memory entry and choose one operation:
- UPDATE: the entry contains information that the new information corrects, changes or makes outdated; rewrite the entry so that it is consistent with the new information.
- DELETE: the entry is entirely contradicted by the new information.
- NONE: the entry is unaffected.

Return only JSON: {{"op": "UPDATE" or "DELETE" or "NONE", "text": "rewritten entry if UPDATE, else empty"}}"""
