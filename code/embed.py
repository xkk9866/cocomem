"""Dense embeddings (bge-base-en-v1.5) with a persistent cache."""
import hashlib
import os
import sqlite3
import threading

import numpy as np

MODEL = "/data/yaokang/corrmem/models/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE = os.environ.get("CM_CACHE", "/data/yaokang/corrmem/cache") + "/emb.sqlite"


class Embedder:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = Embedder()
        return cls._inst

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        dev = os.environ.get("CM_EMB_DEVICE", "cuda")
        self.m = SentenceTransformer(MODEL, device=dev)
        if dev == "cuda":
            self.m.half()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        self.db = sqlite3.connect(CACHE, check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v BLOB)")
        self.lock = threading.Lock()

    def encode(self, texts, query=False):
        if len(texts) == 0:
            return np.zeros((0, 768), dtype=np.float32)
        texts = [(QUERY_PREFIX + t if query else t)[:4000] for t in texts]
        keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        out = [None] * len(texts)
        with self.lock:
            for i, k in enumerate(keys):
                r = self.db.execute("SELECT v FROM e WHERE k=?", (k,)).fetchone()
                if r:
                    out[i] = np.frombuffer(r[0], dtype=np.float16).astype(np.float32)
        miss = [i for i, v in enumerate(out) if v is None]
        if miss:
            with self.lock:
                vecs = self.m.encode([texts[i] for i in miss], batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
                for i, v in zip(miss, vecs):
                    out[i] = v.astype(np.float32)
                    self.db.execute("INSERT OR REPLACE INTO e VALUES (?,?)", (keys[i], v.astype(np.float16).tobytes()))
                self.db.commit()
        return np.stack(out)
