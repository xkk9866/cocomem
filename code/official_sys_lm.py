"""LangMem (langmem==0.0.29) through its own memory store manager: the LLM extracts memories with tool calls
(insert / update; deletion off, the library default) against a langgraph store; retrieval is the manager's own search."""
import os
import re

from official_sys import EMB, MODEL, Sys, base_url, cut

_ENC = None


def embed(texts):
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer(EMB, device="cpu")
    return _ENC.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()


class LangMem(Sys):
    name = "langmem"
    K = 100

    def __init__(self, tag, persons=None, state=None):
        super().__init__(tag, persons)
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        from langchain.chat_models import init_chat_model
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager
        self.ns = ("memories", "u")
        self.store = InMemoryStore(index={"dims": 768, "embed": embed, "fields": ["content"]})
        llm = init_chat_model(f"openai:{MODEL}", base_url=base_url(tag), temperature=0.0)
        self.mgr = create_memory_store_manager(llm, namespace=self.ns, store=self.store)
        for key, val in state or []:
            self.store.put(self.ns, key, val)

    def add_session(self, s):
        msgs = []
        for i, t in enumerate(s["turns"]):
            role = t["role"]
            if self.persons:
                msgs.append({"role": "user" if role == self.persons[0] else "assistant", "content": f"{role}: {t['content']}"})
            else:
                msgs.append({"role": role, "content": t["content"]})
        msgs[0] = dict(msgs[0], content=f"[{s['date']}] {msgs[0]['content']}")
        self.mgr.invoke({"messages": msgs})

    def _all(self):
        return [(x.key, x.value) for x in self.store.search(self.ns, limit=100000)]

    def search(self, q, budget_chars):
        r = self.mgr.search(query=q, limit=self.K)
        txt = lambda v: v.get("content", str(v)) if isinstance(v, dict) else str(getattr(v, "content", v))
        return cut([txt(x.value) for x in r], budget_chars)

    def snapshot(self):
        return self._all()

    def items(self):
        txt = lambda v: v.get("content", str(v)) if isinstance(v, dict) else str(getattr(v, "content", v))
        return [txt(v) for _, v in self._all()]
