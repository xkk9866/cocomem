"""Async client for a local vLLM OpenAI-compatible server, with a persistent
sqlite cache and token accounting. All LLM calls in the project go through
`LLM.chat`, so every method's cost is measured the same way."""
import asyncio
import contextlib
import glob
import hashlib
import itertools
import json
import os
import re
import sqlite3
import threading
import time

from openai import AsyncOpenAI

CACHE_DIR = os.environ.get("CM_CACHE", "/data/yaokang/corrmem/cache")
NOCACHE = bool(os.environ.get("CM_NOCACHE"))      # latency measurement: every call goes to the server


class ContextOverflow(Exception):
    """The request does not fit the server's context window (servers on shared GPUs run with a reduced window;
    callers defer such requests to a server with the full window instead of retrying)."""


def _overflow(e):
    s = str(e)
    return ("maximum context length" in s or "max_model_len" in s or "longer than the maximum" in s
            or "exceeds the model" in s or "input tokens" in s and "context" in s)


LOG_DIR = os.environ.get("CM_LOGS", "/data/yaokang/corrmem/logs")
_KVLINE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_GATES = {}


class _KVGate:
    """Admission control by the KV-cache capacity of one server.

    A vLLM server admits a request as soon as its prompt fits and preempts running requests when the cache fills up
    later.  On the servers of this project (small caches on shared cards) preempted requests came back with corrupted
    output (repetition of single tokens, empty answers), which silently turns into wrong memory text or wrong
    answers.  Every request therefore reserves its prompt plus its full output window before it is sent, first come
    first served, so that the requests in flight always fit the cache and the server never preempts."""

    def __init__(self, cap):
        self.cap, self.used, self.q = cap, 0, []

    @staticmethod
    def need(messages, max_tokens):
        chars = len(messages) if isinstance(messages, str) else sum(len(m.get("content") or "") for m in messages)
        return int(chars / 2.5) + 64 + max_tokens          # ids and dates run at 2.5-3 characters per token

    async def acquire(self, n):
        n = min(n, self.cap)
        if not self.q and self.used + n <= self.cap:
            self.used += n
            return n
        fut = asyncio.get_running_loop().create_future()
        self.q.append((n, fut))
        await fut
        return n

    def release(self, n):
        self.used -= n
        while self.q and (self.q[0][1].cancelled() or self.used + self.q[0][0] <= self.cap):
            m, fut = self.q.pop(0)
            if not fut.cancelled():
                self.used += m
                fut.set_result(None)


def kv_gate(port):
    """The gate of the server on `port` (None when its capacity is unknown or CM_KV_GATE=0).  The capacity is read
    from the server's startup log; 5% is left unreserved."""
    if os.environ.get("CM_KV_GATE", "1") == "0":
        return None
    if port not in _GATES:
        cap, newest = None, -1.0
        for f in glob.glob(os.path.join(LOG_DIR, f"vllm_*_{port}.log")):
            try:
                mt = os.path.getmtime(f)
                if mt < newest:
                    continue
                with open(f, errors="ignore") as s:
                    for i, line in enumerate(s):
                        m = _KVLINE.search(line)
                        if m:
                            cap, newest = int(int(m.group(1).replace(",", "")) * float(m.group(2)) * 0.95), mt
                            break
                        if i > 4000:
                            break
            except OSError:
                pass
        _GATES[port] = _KVGate(cap) if cap else None
        print(f"[llm] port {port}: KV gate {'off (capacity unknown)' if not cap else f'{cap} tokens'}", flush=True)
    return _GATES[port]


class Usage:
    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, p, c):
        self.calls += 1
        self.prompt_tokens += p
        self.completion_tokens += c

    def as_dict(self):
        return {"calls": self.calls, "in": self.prompt_tokens, "out": self.completion_tokens}

    def __iadd__(self, o):
        self.calls += o.calls
        self.prompt_tokens += o.prompt_tokens
        self.completion_tokens += o.completion_tokens
        return self


class _Cache:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        # Regenerable inference cache: avoid a disk fsync for every token-response entry.
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self.db.commit()


class LLM:
    def __init__(self, model, port, concurrency=96, cache_name=None, think=False):
        self.logical_model = model
        is_student = model.startswith("tnet")
        if is_student:
            model += os.environ.get("CM_STUDENT_SUFFIX", "")
        self.model = model
        self.client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY", timeout=1800)
        replica_ports = os.environ.get("CM_Q8_REPLICAS", "") if model == "qwen3-8b" else ""
        ports = list(dict.fromkeys(int(p.strip()) for p in replica_ports.split(",") if p.strip()))
        clients = [self.client if p == port else AsyncOpenAI(
            base_url=f"http://localhost:{p}/v1", api_key="EMPTY", timeout=1800) for p in ports]
        self._clients = itertools.cycle(clients or [self.client])
        self._thinking_clients = itertools.cycle(clients or [self.client])
        self._gate = {id(c): kv_gate(p) for c, p in zip(clients or [self.client], ports or [port])}
        self.sem = asyncio.Semaphore(concurrency)
        cache_dir = CACHE_DIR if is_student else os.environ.get("CM_FIXED_CACHE", CACHE_DIR)
        self.cache = _Cache(os.path.join(cache_dir, (cache_name or model) + ".sqlite"))
        self.think = think

    @contextlib.asynccontextmanager
    async def reserved(self, client, messages, max_tokens):
        """Hold the KV-cache reservation of one request sent to `client` directly (callers that need logprobs)."""
        gate = self._gate.get(id(client))
        held = await gate.acquire(gate.need(messages, max_tokens)) if gate else 0
        try:
            yield
        finally:
            if gate:
                gate.release(held)

    def next_client(self, think=False):
        """Balance reasoning and short requests independently across identical replicas.

        Paired short/reasoning proposals must not pin all long jobs to one GPU.
        """
        return next(self._thinking_clients if think else self._clients)

    async def chat(self, messages, max_tokens=1024, temperature=0.0, usage=None, think=None, stop=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        think = self.think if think is None else think
        key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, temperature, think, stop]).encode()).hexdigest()
        hit = None if NOCACHE else self.cache.get(key)
        if hit is not None:
            # cached calls are still charged so that cost reflects the method, not the cache
            if usage is not None:
                usage.add(hit["p"], hit["c"])
            return hit["text"]
        async with self.sem:
            client = self.next_client(think=think)
            gate = self._gate.get(id(client))
            held = await gate.acquire(gate.need(messages, max_tokens)) if gate else 0
            try:
                for attempt in range(6):
                    try:
                        r = await client.chat.completions.create(
                            model=self.model, messages=messages, max_tokens=max_tokens,
                            temperature=temperature, stop=stop,
                            extra_body={"chat_template_kwargs": {"enable_thinking": think}})
                        break
                    except Exception as e:  # server busy / transient
                        if _overflow(e):
                            raise ContextOverflow(str(e)[:300]) from None
                        if attempt == 5:
                            raise
                        await asyncio.sleep(5 * (attempt + 1))
            finally:
                if gate:
                    gate.release(held)
        text = r.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        p, c = r.usage.prompt_tokens, r.usage.completion_tokens
        self.cache.put(key, {"text": text, "p": p, "c": c})
        if usage is not None:
            usage.add(p, c)
        return text


    async def complete(self, prompt, max_tokens=64, usage=None):
        """Raw greedy completion of a pre-formatted prompt (used to continue a fixed answer prefix)."""
        key = hashlib.sha256(json.dumps([self.model, "complete", prompt, max_tokens]).encode()).hexdigest()
        hit = None if NOCACHE else self.cache.get(key)
        if hit is not None:
            if usage is not None:
                usage.add(hit["p"], hit["c"])
            return hit["text"]
        async with self.sem:
            client = self.next_client()
            gate = self._gate.get(id(client))
            held = await gate.acquire(gate.need(prompt, max_tokens)) if gate else 0
            try:
                for attempt in range(6):
                    try:
                        r = await client.completions.create(model=self.model, prompt=prompt, max_tokens=max_tokens,
                                                            temperature=0.0)
                        break
                    except Exception as e:
                        if _overflow(e):
                            raise ContextOverflow(str(e)[:300]) from None
                        if attempt == 5:
                            raise
                        await asyncio.sleep(5 * (attempt + 1))
            finally:
                if gate:
                    gate.release(held)
        text = r.choices[0].text or ""
        p, c = r.usage.prompt_tokens, r.usage.completion_tokens
        self.cache.put(key, {"text": text, "p": p, "c": c})
        if usage is not None:
            usage.add(p, c)
        return text


def parse_json(text, default=None):
    """Extract the first JSON object/array from an LLM response."""
    if text is None:
        return default
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        if i < 0:
            continue
        depth, instr, esc = 0, False, False
        for j in range(i, len(t)):
            ch = t[j]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    s = t[i:j + 1]
                    for cand in (s, s.replace('"""', '"')):
                        try:
                            return json.loads(cand, strict=False)   # tolerate raw newlines in strings
                        except Exception:
                            pass
                    break
    return default


async def gather_limited(coros, limit=None, desc=None, every=50):
    """asyncio.gather with progress printing."""
    n = len(coros)
    done = 0
    t0 = time.time()
    results = [None] * n

    async def wrap(i, c):
        nonlocal done
        results[i] = await c
        done += 1
        if desc and (done % every == 0 or done == n):
            el = time.time() - t0
            print(f"[{desc}] {done}/{n} {el:.0f}s eta {el / done * (n - done):.0f}s", flush=True)

    await asyncio.gather(*(wrap(i, c) for i, c in enumerate(coros)))
    return results
