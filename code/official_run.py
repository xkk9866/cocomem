"""Run the official baseline systems (official_sys.py) on the correction tasks and store what they retrieve.
One process per memory (the official libraries are synchronous); the shared LLM server is reached through llm_proxy.
Stage B (official_read.py, main environment) lets the shared reader answer from the stored contexts.
python official_run.py --system mem0 --task lme --splits test200 --procs 24"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import shutil
import glob
import urllib.request

TASKS = "/data/yaokang/corrmem/bl/tasks"
OUT = "/data/yaokang/corrmem/bl/out"
BUDGET = 4000 * 4          # the same 4000-token context budget as every other method (len // 4 tokens)


def usage(tag):
    port = int(os.environ.get("BL_PROXY", 18440))
    return json.load(urllib.request.urlopen(f"http://localhost:{port}/_usage/{tag}", timeout=60))


def udiff(a, b):
    return {k: b[k] - a[k] for k in b}


def safe_add(m, s, errs):
    try:
        m.add_session(s)
    except Exception as e:
        errs.append(f"{s['sid']}: {type(e).__name__}: {str(e)[:200]}")


def finish(res, path, errs, rt):
    """store the result; the on-disk stores of the memory are not needed any more.  A run in which the LLM server failed
    (HTTP 5xx) is not a result of the system: it is not stored and the memory is re-run."""
    from official_sys import STORE
    for d in glob.glob(f"{STORE}/{rt}_*"):
        shutil.rmtree(d, ignore_errors=True)
    if any("Error code: 5" in e or "Connection" in e for e in errs):
        raise RuntimeError("LLM server error: " + errs[0])
    res["errors"] = errs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"))
    return "ok" if not errs else f"ok(errors={len(errs)})"


def lme(job):
    sysname, t, tag = job
    from official_sys import SYSTEMS
    S = SYSTEMS[sysname]
    path = f"{OUT}/{tag}/{sysname}/{t['qid']}.json"
    if os.path.exists(path):
        return "cached"
    errs = []
    rt = f"{sysname}_{t['qid']}"
    try:
        t0 = time.time()
        m = S(rt + "_b")
        for s in t["sessions"]:
            safe_add(m, s, errs)
        bt = time.time() - t0
        res = {"qid": t["qid"], "system": sysname, "build": {"usage": usage(rt + "_b"), "time": bt}, "ctx": {}, "items": {}, "repair": {}}
        state = m.snapshot()
        res["items"]["S0"] = m.items()
        for sc, key, q, gold, jt in t["queries"]:
            if sc == "S0":
                res["ctx"][f"S0:{key}"] = m.search(q, BUDGET)
        for sc in ("S1", "S2"):
            if sc not in t["corr"]:
                continue
            m2 = S(f"{rt}_{sc}", state=state)
            t0 = time.time()
            safe_add(m2, t["corr"][sc], errs)
            res["repair"][sc] = {"usage": usage(f"{rt}_{sc}"), "time": time.time() - t0}
            res["items"][sc] = m2.items()
            for s2, key, q, gold, jt in t["queries"]:
                if s2 == sc:
                    res["ctx"][f"{sc}:{key}"] = m2.search(q, BUDGET)
        return finish(res, path, errs, rt)
    except Exception:
        open(f"{OUT}/{tag}/errors.log", "a").write(f"{t['qid']}\n{traceback.format_exc()}\n")
        return "fail"


def locomo(job):
    sysname, t, tag = job
    from official_sys import SYSTEMS
    S = SYSTEMS[sysname]
    path = f"{OUT}/{tag}/{sysname}/{t['cid']}.json"
    if os.path.exists(path):
        return "cached"
    errs = []
    rt = f"{sysname}_{t['cid']}"
    try:
        persons = t["persons"]
        t0 = time.time()
        m = S(rt + "_b", persons=persons)
        for s in t["sessions"]:
            safe_add(m, s, errs)
        res = {"cid": t["cid"], "system": sysname, "build": {"usage": usage(rt + "_b"), "time": time.time() - t0},
               "ctx": {}, "reg": {}, "steps": [], "items": {}}
        res["items"]["S0"] = m.items()
        Q = t["queries"]
        for sc, key, q, gold, jt, cat in Q:
            if sc == "S0":
                res["ctx"][f"S0:{key}"] = m.search(q, BUDGET)
        # sequential corrections on the SAME memory (the interference test); after every step the corrections made so far and a
        # fixed panel of unaffected questions are re-asked (recurrence test)
        m = S(rt + "_c", persons=persons, state=m.snapshot())
        prev = usage(rt + "_c")
        t0 = time.time()
        for step, s in enumerate(t["corr"]):
            tt = time.time()
            safe_add(m, s, errs)
            cur = usage(rt + "_c")
            res["steps"].append({"usage": udiff(prev, cur), "time": time.time() - tt})
            prev = cur
            for sc, key, q, gold, jt, cat in Q:
                if sc != "S1":
                    continue
                if key.startswith("a") and int(key[1:]) <= step or key.startswith("u") and int(key[1:]) < 5:
                    res["reg"][f"{step}:{key}"] = m.search(q, BUDGET)
        res["repair"] = {"usage": usage(rt + "_c"), "time": time.time() - t0}
        res["items"]["S1"] = m.items()
        for sc, key, q, gold, jt, cat in Q:
            if sc == "S1":
                res["ctx"][f"S1:{key}"] = m.search(q, BUDGET)
        return finish(res, path, errs, rt)
    except Exception:
        open(f"{OUT}/{tag}/errors.log", "a").write(f"{t['cid']}\n{traceback.format_exc()}\n")
        return "fail"


def init():
    sys.stdout = open(os.devnull, "w")          # A-Mem prints every prompt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--task", default="lme")
    ap.add_argument("--splits", default="test200")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.tag}/{a.system}", exist_ok=True)
    rows = json.load(open(f"{TASKS}/{a.task}.json"))
    if a.task == "lme":
        rows = [r for r in rows if r["split"] in a.splits.split(",")]
    if a.limit:
        rows = rows[:a.limit]
    fn = lme if a.task == "lme" else locomo
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.procs, initializer=init) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(fn, [(a.system, r, a.tag) for r in rows])):
            res.append(r)
            if (i + 1) % 10 == 0:
                print(a.system, a.task, i + 1, len(rows), f"{time.time() - t0:.0f}s", flush=True, file=sys.stderr)
    print(a.system, a.task, {r: res.count(r) for r in set(res)}, f"{time.time() - t0:.0f}s", file=sys.stderr)
