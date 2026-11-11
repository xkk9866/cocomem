"""Adapters that drive the OFFICIAL code of the baseline memory systems through one interface.
Nothing of a baseline's own logic is re-implemented: the adapter only (i) hands a session to the system's own
write path (`add` / `add_note` / `manage`), (ii) calls the system's own retrieval, and (iii) snapshots / restores the
stored memories so that the two correction branches (S1, S2) start from the same memory.
LLM: the shared Qwen3-8B-AWQ through llm_proxy (thinking off, tokens counted per tag); embeddings: bge-base on CPU for all.
  mem0     mem0ai==1.0.2 (the ADD / UPDATE / DELETE pipeline of the Mem0 paper, arXiv 2504.19413)
  mem0v2   current mem0ai (additive extraction)
  amem     agiresearch/A-mem (WujiangXu/A-mem, paper code)
  langmem  langmem==0.0.29 memory store manager (LangChain)"""
import copy
import datetime as dt
import os
import re
import sys

EMB = "/data/yaokang/corrmem/models/bge-base-en-v1.5"
PROXY = int(os.environ.get("BL_PROXY", 18440))
MODEL = os.environ.get("BL_MODEL", "qwen3-8b")
STORE = "/data/yaokang/corrmem/bl/store"


def base_url(tag):
    return f"http://localhost:{PROXY}/{tag}/v1"


def ts(date):
    """'2023/05/21 (Sun) 10:00' -> '202305211000' (A-Mem's timestamp format)"""
    return dt.datetime.strptime(date[:10] + date[-6:], "%Y/%m/%d %H:%M").strftime("%Y%m%d%H%M")


def iso(date):
    return dt.datetime.strptime(date[:10] + date[-6:], "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M")


def rounds(s, persons=None):
    """session -> list of rounds (two consecutive turns) as plain text"""
    out = []
    T = s["turns"]
    nm = lambda t: t["role"].capitalize() if not persons else t["role"]
    for i in range(0, len(T), 2):
        out.append("\n".join(f"{nm(t)}: {t['content'].strip()}" for t in T[i:i + 2])[:3000])
    return out


class Sys:
    name = "base"

    def __init__(self, tag, persons=None):
        self.tag, self.persons = tag, persons

    def add_session(self, s):
        raise NotImplementedError

    def search(self, q, budget_chars):
        raise NotImplementedError

    def snapshot(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError


def cut(lines, budget_chars):
    out, used = [], 0
    for l in lines:
        if used + len(l) + 1 > budget_chars:
            continue
        out.append(l)
        used += len(l) + 1
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ Mem0
class Mem0(Sys):
    name = "mem0"
    K = 100

    def __init__(self, tag, persons=None, state=None):
        super().__init__(tag, persons)
        os.environ["MEM0_TELEMETRY"] = "False"
        from mem0 import Memory
        import mem0.memory.main as mm
        os.makedirs(f"{STORE}/{tag}", exist_ok=True)
        mm.mem0_dir = f"{STORE}/{tag}"          # mem0 keeps a telemetry/migration store under this directory: one per memory
        self.uid = "u"
        cfg = {"version": "v1.1",
               "llm": {"provider": "openai", "config": {"model": MODEL, "api_key": "EMPTY", "openai_base_url": base_url(tag)}},
               "vector_store": {"provider": "qdrant", "config": {"collection_name": "c" + re.sub(r"\W", "_", tag),
                                                                  "embedding_model_dims": 768, "path": f"{STORE}/{tag}/q", "on_disk": False}},
               "embedder": {"provider": "huggingface", "config": {"model": EMB, "embedding_dims": 768, "model_kwargs": {"device": "cpu"}}},
               "history_db_path": f"{STORE}/{tag}/h.db"}
        self.m = Memory.from_config(cfg)
        for it in state or []:
            self.m.add(it["memory"], user_id=self.uid, infer=False, metadata={"date": it["date"]})

    def add_session(self, s):
        msgs = []
        for t in s["turns"]:
            role = t["role"]
            if self.persons:      # two named speakers: the first plays the user, the second the assistant
                msgs.append({"role": "user" if role == self.persons[0] else "assistant", "content": f"{role}: {t['content']}"})
            else:
                msgs.append({"role": role, "content": t["content"]})
        self.m.add(msgs, user_id=self.uid, metadata={"date": s["date"]})

    def _all(self):
        r = self.m.get_all(user_id=self.uid, limit=100000)
        r = r["results"] if isinstance(r, dict) else r
        return [{"memory": x["memory"], "date": (x.get("metadata") or {}).get("date", "")} for x in r]

    def search(self, q, budget_chars):
        r = self.m.search(q, user_id=self.uid, limit=self.K)
        r = r["results"] if isinstance(r, dict) else r
        return cut([f"[{(x.get('metadata') or {}).get('date', '')}] {x['memory']}" for x in r], budget_chars)

    def snapshot(self):
        return self._all()

    def items(self):
        return [f"[{x['date']}] {x['memory']}" for x in self._all()]


class Mem0v2(Mem0):
    """current mem0ai: additive extraction; entity scoping only through `filters`, result count `top_k`"""
    name = "mem0v2"

    def _all(self):
        r = self.m.get_all(filters={"user_id": self.uid}, top_k=100000)
        r = r["results"] if isinstance(r, dict) else r
        return [{"memory": x["memory"], "date": (x.get("metadata") or {}).get("date", "")} for x in r]

    def search(self, q, budget_chars):
        r = self.m.search(q, filters={"user_id": self.uid}, top_k=self.K)
        r = r["results"] if isinstance(r, dict) else r
        return cut([f"[{(x.get('metadata') or {}).get('date', '')}] {x['memory']}" for x in r], budget_chars)


# ------------------------------------------------------------------------------------------------ A-Mem
class AMem(Sys):
    name = "amem"
    K = 10
    _model = None

    def __init__(self, tag, persons=None, state=None):
        super().__init__(tag, persons)
        sys.path.insert(0, "/data/yaokang/corrmem/bl/A-mem")
        os.environ["OPENAI_BASE_URL"] = base_url(tag)
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        import memory_layer as ml
        self.ml = ml
        from sentence_transformers import SentenceTransformer
        if AMem._model is None:
            AMem._model = SentenceTransformer(EMB, device="cpu")
        # one shared encoder object for every retriever the system creates (its consolidation step re-creates the
        # retriever and falls back to the default MiniLM name; the shared bge-base keeps the embedder identical everywhere)
        ml.SentenceTransformer = lambda *a, **k: AMem._model
        self.s = ml.AgenticMemorySystem(model_name=EMB, llm_backend="openai", llm_model=MODEL, evo_threshold=100)
        if state:
            self.s.memories = copy.deepcopy(state["memories"])
            self.s.retriever.corpus = list(state["corpus"])
            self.s.retriever.embeddings = None if state["emb"] is None else state["emb"].copy()
            self.s.retriever.document_ids = dict(state["ids"])
            self.s.evo_cnt = state["evo"]

    def add_session(self, s):
        if os.environ.get("BL_AMEM_SESSION"):
            # one note per session (LongMemEval sessions have ~500 turns per memory: per-turn notes cost >2 GPU-hours per memory)
            txt = "\n".join(f"{t['role'].capitalize() if not self.persons else t['role']}: {t['content'].strip()}" for t in s["turns"])
            self.s.add_note(txt[:12000], time=ts(s["date"]))
            return
        for r in rounds(s, self.persons):
            self.s.add_note(r, time=ts(s["date"]))

    def search(self, q, budget_chars):
        txt = self.s.find_related_memories_raw(q, k=self.K)
        return txt[:budget_chars] if isinstance(txt, str) else ""

    def snapshot(self):
        r = self.s.retriever
        return {"memories": copy.deepcopy(self.s.memories), "corpus": list(r.corpus),
                "emb": None if r.embeddings is None else r.embeddings.copy(), "ids": dict(r.document_ids), "evo": self.s.evo_cnt}

    def items(self):
        return [f"[{m.timestamp}] {m.content} (context: {m.context}; keywords: {m.keywords}; tags: {m.tags})"
                for m in self.s.memories.values()]


SYSTEMS = {"mem0": Mem0, "mem0v2": Mem0v2, "amem": AMem}
try:
    from official_sys_lm import LangMem
    SYSTEMS["langmem"] = LangMem
except Exception:
    pass
