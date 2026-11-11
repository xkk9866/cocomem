"""End-to-end baseline memory systems, re-implemented on the shared backbone (Qwen3-8B-AWQ),
shared retriever (bge-base) and shared reader/budget.  Prompts are the official ones where the
original repositories provide them (Mem0, A-Mem)."""
import asyncio
import copy
import datetime as dt
import json
import re

import numpy as np

import mem0_prompts as M0
from cocomem import emb, ntok
from data import format_session
from llm import Usage, parse_json


ROLE = {"user": "User", "assistant": "Assistant"}


def iso(date):
    return dt.datetime.strptime(date[:10], "%Y/%m/%d").strftime("%Y-%m-%d")


class Base:
    name = "base"

    def __init__(self, llm):
        self.llm = llm
        self.items = []          # retrievable memory items: dict(text, date, ...)
        self.core = None         # always-in-context memory (profile/portrait), if any
        self.usage = {"write": Usage(), "repair": Usage(), "read": Usage()}

    async def add_session(self, s, usage):
        raise NotImplementedError

    async def build(self, sessions):
        for s in sessions:
            await self.add_session(s, self.usage["write"])

    async def correct(self, session, strategy=None, usage=None):
        await self.add_session(session, usage or self.usage["repair"])
        return {}

    def item_text(self, it):
        return it["text"]

    async def retrieve(self, question, budget=4000):
        items = [it for it in self.items if it.get("text")]
        lines = []
        used = ntok(self.core) if self.core else 0
        chosen = []
        if items:
            E = await emb([self.item_text(it) for it in items])
            q = (await emb([question], query=True))[0]
            for i in np.argsort(-(E @ q)):
                t = ntok(items[i]["text"]) + 12
                if used + t > budget:
                    continue
                chosen.append(items[i])
                used += t
        chosen.sort(key=lambda z: z["date"])
        if self.core:
            lines.append(f"[User profile] {self.core}")
        for it in chosen:
            lines.append(f"[{it['date']}] {it['text']}")
        return chosen, "\n".join(lines)

    def to_json(self):
        return {"items": self.items, "core": self.core, "extra": self.extra_state(),
                "usage": {k: v.as_dict() for k, v in self.usage.items()}}

    def extra_state(self):
        return {}

    def load_extra(self, d):
        pass

    @classmethod
    def from_json(cls, llm, d):
        m = cls(llm)
        m.items = copy.deepcopy(d["items"])
        m.core = d["core"]
        m.load_extra(copy.deepcopy(d.get("extra", {})))
        return m

    def clone(self):
        return type(self).from_json(self.llm, self.to_json())

    def size_tokens(self):
        return {"items": sum(ntok(i["text"]) for i in self.items), "core": ntok(self.core) if self.core else 0}


class RawRAG(Base):
    """No compression: round-level (user+assistant) values, as recommended by LongMemEval."""
    name = "rawrag"

    async def add_session(self, s, usage):
        turns = s["turns"]
        for i in range(0, len(turns), 2):
            rnd = turns[i:i + 2]
            text = " ".join(f"{ROLE.get(t['role'], t['role'])}: {t['content'].strip()}" for t in rnd)
            self.items.append({"text": text[:3000], "date": s["date"], "sid": s["sid"]})

    def item_text(self, it):
        return it["text"][:2000]


MB_SUMMARY = """Please summarize the events and key information in the following conversation between a user and an AI assistant, especially details about the user (personal information, events, plans, preferences, quantities, names, dates) and any specific information the assistant provided. Be concise (at most 120 words).

{session}

Summary:"""

MB_PORTRAIT = """You maintain a global portrait of a user for a long-term AI companion (MemoryBank). Given the current user portrait and the summary of the newest conversation (dated {date}), update the portrait: the user's personal information, life situation, personality, preferences and anything that changed. Keep specific values. At most 250 words.

Current portrait:
{portrait}

Newest conversation summary:
{summary}

Updated user portrait:"""


class MemoryBank(Base):
    """MemoryBank-style hierarchical memory: per-session event summaries + an incrementally updated
    global user portrait (always in context)."""
    name = "membank"

    async def summarize(self, s, usage):
        return await self.llm.chat(MB_SUMMARY.format(session=format_session(s["turns"], s["date"], number=False)),
                                   max_tokens=300, usage=usage)

    async def add_session(self, s, usage, summary=None):
        summary = summary or await self.summarize(s, usage)
        self.items.append({"text": summary.strip(), "date": s["date"], "sid": s["sid"]})
        self.core = (await self.llm.chat(MB_PORTRAIT.format(date=s["date"], portrait=self.core or "(empty)",
                                                            summary=summary), max_tokens=500, usage=usage)).strip()

    async def build(self, sessions):
        sums = await asyncio.gather(*(self.summarize(s, self.usage["write"]) for s in sessions))
        for s, sm in zip(sessions, sums):
            await self.add_session(s, self.usage["write"], summary=sm)


class Mem0(Base):
    """Mem0 (official FACT_RETRIEVAL_PROMPT and DEFAULT_UPDATE_MEMORY_PROMPT; top-5 similar memories per fact)."""
    name = "mem0"

    async def extract(self, s, usage):
        sys = re.sub(r"Today's date is [0-9-]+\.", f"Today's date is {iso(s['date'])}.", M0.FACT_RETRIEVAL_PROMPT)
        conv = "\n".join(f"{t['role']}: {t['content']}" for t in s["turns"])
        r = await self.llm.chat([{"role": "system", "content": sys},
                                 {"role": "user", "content": f"Input:\n{conv}"}], max_tokens=600, usage=usage)
        js = parse_json(r, {})
        facts = js.get("facts", []) if isinstance(js, dict) else []
        return [f for f in facts if isinstance(f, str) and f.strip()]

    async def update(self, s, facts, usage):
        if not facts:
            return
        cand = set()
        if self.items:
            E = await emb([it["text"] for it in self.items])
            Q = await emb(facts, query=True)
            for i in range(len(facts)):
                for j in np.argsort(-(Q[i] @ E.T))[:5]:
                    cand.add(int(j))
        cand = sorted(cand)
        idmap = {str(k): j for k, j in enumerate(cand)}
        old = [{"id": str(k), "text": self.items[j]["text"]} for k, j in enumerate(cand)]
        prompt = M0.get_update_memory_messages(old, facts)
        r = await self.llm.chat(prompt, max_tokens=1500, usage=usage)
        js = parse_json(r, {})
        mem = js.get("memory", []) if isinstance(js, dict) else []
        dele = set()
        added = False
        for m in mem:
            if not isinstance(m, dict):
                continue
            ev = str(m.get("event", "")).upper()
            txt = str(m.get("text", "")).strip()
            mid = str(m.get("id", ""))
            if ev == "ADD" and txt:
                self.items.append({"text": txt, "date": s["date"]})
                added = True
            elif ev == "UPDATE" and mid in idmap and txt:
                self.items[idmap[mid]] = {"text": txt, "date": s["date"]}
            elif ev == "DELETE" and mid in idmap:
                dele.add(idmap[mid])
        if not mem and not added:   # malformed output: fall back to ADD (as Mem0 does when no memory exists)
            for f in facts:
                self.items.append({"text": f, "date": s["date"]})
        self.items = [it for k, it in enumerate(self.items) if k not in dele]

    async def add_session(self, s, usage, facts=None):
        facts = facts if facts is not None else await self.extract(s, usage)
        await self.update(s, facts, usage)

    async def build(self, sessions):
        facts = await asyncio.gather(*(self.extract(s, self.usage["write"]) for s in sessions))
        for s, f in zip(sessions, facts):
            await self.add_session(s, self.usage["write"], facts=f)


AMEM_ANALYZE = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context":
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """

AMEM_EVOLVE = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''


class AMem(Base):
    """A-Mem (official analysis and evolution prompts). One note per round (user+assistant)."""
    name = "amem"

    def item_text(self, it):
        return f"{it['text'][:1500]} context: {it.get('context', '')} keywords: {', '.join(it.get('keywords', []))} tags: {', '.join(it.get('tags', []))}"

    async def analyze(self, content, usage):
        r = await self.llm.chat(AMEM_ANALYZE + content[:6000], max_tokens=400, usage=usage)
        js = parse_json(r.replace("True", "true").replace("False", "false"), {})
        js = js if isinstance(js, dict) else {}
        return {"keywords": [str(x) for x in js.get("keywords", [])][:10], "context": str(js.get("context", "")),
                "tags": [str(x) for x in js.get("tags", [])][:10]}

    @staticmethod
    def rounds(s):
        out = []
        for i in range(0, len(s["turns"]), 2):
            out.append(" ".join(f"{ROLE.get(t['role'], t['role'])}: {t['content'].strip()}"
                                for t in s["turns"][i:i + 2])[:3000])
        return out

    async def evolve(self, note, usage, k=5):
        if self.items:
            E = await emb([self.item_text(it) for it in self.items])
            q = (await emb([note["text"][:2000]], query=True))[0]
            idx = [int(j) for j in np.argsort(-(E @ q))[:k]]
            nb = "\n".join(f"memory index:{n}\ttalk start time:{self.items[j]['date']}\tmemory content: {self.items[j]['text'][:600]}\t"
                           f"memory context: {self.items[j].get('context', '')}\tmemory keywords: {self.items[j].get('keywords', [])}\t"
                           f"memory tags: {self.items[j].get('tags', [])}" for n, j in enumerate(idx))
            r = await self.llm.chat(AMEM_EVOLVE.format(context=note["context"], content=note["text"][:2000],
                                                       keywords=note["keywords"], nearest_neighbors_memories=nb,
                                                       neighbor_number=len(idx)), max_tokens=800, usage=usage)
            js = parse_json(r.replace("True", "true").replace("False", "false"), {})
            if isinstance(js, dict) and js.get("should_evolve"):
                for a in js.get("actions") or []:
                    if a == "strengthen":
                        note["links"] = [idx[int(c)] for c in js.get("suggested_connections") or []
                                         if str(c).isdigit() and int(c) < len(idx)]
                        if js.get("tags_to_update"):
                            note["tags"] = [str(x) for x in js["tags_to_update"]]
                    elif a == "update_neighbor":
                        ctxs = js.get("new_context_neighborhood") or []
                        tags = js.get("new_tags_neighborhood") or []
                        for n, j in enumerate(idx):
                            if n < len(ctxs) and ctxs[n]:
                                self.items[j]["context"] = str(ctxs[n])
                            if n < len(tags) and isinstance(tags[n], list) and tags[n]:
                                self.items[j]["tags"] = [str(x) for x in tags[n]]
        self.items.append(note)

    async def add_session(self, s, usage, analyses=None):
        rs = self.rounds(s)
        analyses = analyses or await asyncio.gather(*(self.analyze(r, usage) for r in rs))
        for r, a in zip(rs, analyses):
            await self.evolve({"text": r, "date": s["date"], **a}, usage)

    async def build(self, sessions):
        allr = [(s, r) for s in sessions for r in self.rounds(s)]
        an = await asyncio.gather(*(self.analyze(r, self.usage["write"]) for _, r in allr))
        k = 0
        for s in sessions:
            n = len(self.rounds(s))
            await self.add_session(s, self.usage["write"], analyses=an[k:k + n])
            k += n

    async def retrieve(self, question, budget=4000):
        # A-Mem returns note content plus its context description
        items = self.items
        if not items:
            return [], ""
        E = await emb([self.item_text(it) for it in items])
        q = (await emb([question], query=True))[0]
        used, chosen = 0, []
        for i in np.argsort(-(E @ q)):
            txt = f"{items[i]['text'][:2000]} (context: {items[i].get('context', '')})"
            t = ntok(txt) + 12
            if used + t > budget:
                continue
            chosen.append((items[i]["date"], txt))
            used += t
        chosen.sort()
        return chosen, "\n".join(f"[{d}] {t}" for d, t in chosen)


class Oracle(Base):
    """Oracle retrieval: only the evidence sessions (plus any later session, e.g. corrections), read in full.
    Used as an upper-bound reference and to validate generated corrections."""
    name = "oracle"

    async def add_session(self, s, usage):
        if s.get("evidence") or s["sid"].startswith("corr_"):
            self.items.append({"text": format_session(s["turns"], s["date"], number=False), "date": s["date"]})

    async def retrieve(self, question, budget=4000):
        items = sorted(self.items, key=lambda z: z["date"])
        return items, "\n\n".join(it["text"][:24000] for it in items)


def _when(date):
    return dt.datetime.strptime(date[:10] + date[-6:], "%Y/%m/%d %H:%M")


class GenAgents(Base):
    """Generative Agents memory stream (Park et al., UIST 2023).

    Each session is one observation. Retrieval ranks memories by the paper's sum of recency
    (0.995 per hour), importance (1 to 10) and relevance, each min-max normalized. When the
    accumulated importance reaches 150, the stream adds reflective memories."""
    name = "ga"
    DECAY = 0.995

    def __init__(self, llm):
        super().__init__(llm)
        self.acc_importance = 0

    async def add_session(self, s, usage, observation=None):
        text = observation
        if text is None:
            text = await self.llm.chat(
                "Write one memory an assistant would keep from this conversation: one or two sentences, "
                "with the specific names, numbers and dates.\n\n"
                + format_session(s["turns"], s["date"], number=False)[:6000], max_tokens=180, usage=usage)
        rating = await self.llm.chat(
            "On the scale of 1 to 10, where 1 is purely mundane (brushing teeth, making the bed) and 10 is "
            "extremely poignant (a break up, a college acceptance), rate the poignancy of the following memory. "
            "Reply with one integer.\nMemory: " + text.strip()[:1500], max_tokens=4, usage=usage)
        m = re.search(r"\d+", rating or "")
        importance = min(10, max(1, int(m.group(0)))) if m else 5
        self.items.append({"text": text.strip(), "date": s["date"], "importance": importance, "kind": "observation"})
        self.acc_importance += importance
        if self.acc_importance >= 150 and len(self.items) >= 8:
            await self.reflect(s["date"], usage)
            self.acc_importance = 0

    async def reflect(self, date, usage):
        recent = self.items[-20:]
        joined = "\n".join(f"- {it['text'][:400]}" for it in recent)
        insights = await self.llm.chat(
            "Given the following memories, what 3 high-level insights can you infer about the user? "
            "Write exactly 3 lines, one insight each.\n\n" + joined, max_tokens=300, usage=usage)
        for line in insights.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if len(line) > 20:
                self.items.append({"text": line, "date": date, "importance": 7, "kind": "reflection"})

    async def retrieve(self, question, budget=4000):
        items = [it for it in self.items if it.get("text")]
        if not items:
            return [], ""
        now = max(_when(it["date"]) for it in items)
        E = await emb([it["text"] for it in items])
        q = (await emb([question], query=True))[0]
        rel = E @ q
        rec = np.array([self.DECAY ** max(0.0, (now - _when(it["date"])).total_seconds() / 3600.0) for it in items])
        imp = np.array([it.get("importance", 5) for it in items], float)

        def norm(v):
            lo, hi = float(v.min()), float(v.max())
            return np.ones_like(v) if hi - lo < 1e-6 else (v - lo) / (hi - lo)

        score = norm(rec) + norm(imp) + norm(rel)
        used, chosen = 0, []
        for i in np.argsort(-score):
            t = ntok(items[i]["text"]) + 12
            if used + t > budget:
                continue
            chosen.append(items[i])
            used += t
        chosen.sort(key=lambda z: z["date"])
        return chosen, "\n".join(f"[{it['date']}] {it['text']}" for it in chosen)

    def extra_state(self):
        return {"acc_importance": self.acc_importance}

    def load_extra(self, d):
        self.acc_importance = d.get("acc_importance", 0)


class ReadAgent(Base):
    """ReadAgent (Lee et al., ICML 2024): a one-sentence gist of each page, then lookup.

    Each conversation session is one page. At question time the model reads the gists and names
    the pages to open; those pages are then read in full, within the same budget as every other system."""
    name = "readagent"

    async def add_session(self, s, usage, gist=None):
        raw = format_session(s["turns"], s["date"], number=False)
        if gist is None:
            gist = await self.llm.chat(
                "Write a gist of the following page in one sentence. Keep names, numbers and dates.\n\n" + raw[:5000],
                max_tokens=80, usage=usage)
        self.items.append({"text": raw[:5000], "gist": gist.strip()[:500], "date": s["date"], "sid": s["sid"]})

    async def build(self, sessions):
        gists = await asyncio.gather(*(self.llm.chat(
            "Write a gist of the following page in one sentence. Keep names, numbers and dates.\n\n"
            + format_session(s["turns"], s["date"], number=False)[:5000],
            max_tokens=80, usage=self.usage["write"]) for s in sessions))
        for s, g in zip(sessions, gists):
            await self.add_session(s, self.usage["write"], gist=g)

    def item_text(self, it):
        return it.get("gist") or it["text"][:500]

    async def retrieve(self, question, budget=4000):
        if not self.items:
            return [], ""
        listing = "\n".join(f"({i}) {it['gist']}" for i, it in enumerate(self.items))
        picked = await self.llm.chat(
            "A reader has one gist per page and must choose which pages to open to answer the question. "
            "Reply with the page numbers, most useful first, separated by spaces, and nothing else.\n\n"
            f"Pages:\n{listing[:7000]}\n\nQuestion: {question}\n\nPages:",
            max_tokens=80, usage=self.usage["read"])
        order = []
        for tok in re.findall(r"\d+", picked or ""):
            i = int(tok)
            if 0 <= i < len(self.items) and i not in order:
                order.append(i)
        if not order:
            E = await emb([it["gist"] for it in self.items])
            q = (await emb([question], query=True))[0]
            order = [int(i) for i in np.argsort(-(E @ q))]
        used, chosen, lines = 0, [], []
        for i in order:
            page = self.items[i]["text"]
            t = ntok(page) + 16
            if used + t > budget:
                continue
            chosen.append(self.items[i])
            lines.append(f"[{self.items[i]['date']}] {page}")
            used += t
        return chosen, "\n".join(lines)


SYSTEMS = {"rawrag": RawRAG, "membank": MemoryBank, "mem0": Mem0, "amem": AMem, "oracle": Oracle,
           "ga": GenAgents, "readagent": ReadAgent}
