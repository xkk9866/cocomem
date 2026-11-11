"""GPU-aware job scheduler for the v21 experiments on a shared server whose GPUs are mostly occupied by other users.

Every 30 s the scheduler reads the free memory of each allowed GPU and starts the first ready job (dependencies done,
not running, retry time reached) whose memory slots fit on GPUs that no other job of ours uses.  A job declares one
slot per model server it runs: an 8B compressor alone, an 8B compressor with the 24k window, a 4B student, a judge, a
training process.  Two-slot jobs (compressor + student) are placed on one GPU when it has room for both and on two
GPUs otherwise; the job scripts (j_*.sh) receive the placement in the environment and size every server from the
memory that was free at launch.  Jobs are idempotent (finished work is cached), so a job that fails -- usually because
a neighbour's process grew -- is retried later with back-off.

  python sched21.py                 # setsid nohup ../.venv/bin/python sched21.py > ../logs/sched21_stdout.log 2>&1 &
  touch logs/SCHED21.pause          # no new launches (running jobs continue);  rm to resume
  touch logs/SCHED21.stop           # kill running jobs and exit
State: logs/sched21.state.json, markers logs/J_<job>.done, job logs logs/j_<job>.log, scheduler log logs/sched21.log."""
import json
import os
import signal
import subprocess
import sys
import time

P = os.environ.get("CM_ROOT", "/data/yaokang/corrmem")
CODE, LOGS = f"{P}/code", f"{P}/logs"
NAME = os.environ.get("SCHED_NAME", "sched21")       # sched22.py reuses the scheduler with its own job list and files
UP = NAME.upper()
ALLOWED = [int(x) for x in os.environ.get("SCHED_GPUS", "0,1,2,3,4").split(",")]      # GPUs 5, 6, 7 are off limits
MARGIN = 512                                   # MiB left untouched on every GPU
SETTLE = int(os.environ.get("SCHED_SETTLE", 1200))   # seconds until the servers of a launched job hold all their memory
# memory slots (MiB free at launch) -- measured: Qwen3-8B-AWQ weights 5.7 GiB, 4B student (bnb int4) 2.5 GiB
# (servers take 75% of the free memory, j_common.sh; the slot sizes make that 75% cover the server's footprint)
S8 = 10200       # 8B compressor, reduced window (16k; footprint 7.6 GB); over-long requests are deferred
S8F = 11150      # 8B compressor with the full 24k window (8.6 GB + 2.5 GB headroom, see j_common.sh serve8)
S4 = 5300        # int4 student, 8k window (4.0 GB)
ST = 8500        # QLoRA training of the student (4-bit base, token budget 1024-2048)
SJ = 8600        # Qwen2.5-7B-AWQ judge (12k window);  Llama-3.1-8B-AWQ needs S8
POLL = int(os.environ.get("SCHED_POLL", "8"))  # the cards are saturated by many small jobs: a slot that opens is
                                               # gone within seconds, so the memory is read often
NOTE = 600                                     # seconds between two notes about the memory seen while waiting
NSHARD = int(os.environ.get("SCHED_SHARDS", "3"))

# ------------------------------------------------------------------------------------------------------- jobs
J = []


def job(id_, cmd, after=(), need=(S8,)):
    J.append({"id": id_, "cmd": cmd, "after": list(after), "need": list(need)})


STUD = ["tnet21", "tnet21nd", "tnet21nv", "tnet21nr", "tnet21w"]
ROLE = {"tnet21": "main", "tnet21nd": "nd", "tnet21w": "w", "tnet21nv": "nv", "tnet21nr": "nr"}
PAIRS = "deriv_pairs"
# P1 records: the units of the training / held-out pairs (short prompts, reduced window) in shards, and the profile
# units of every memory the experiments touch (prompts list every claim of the user: full window)
for i in range(NSHARD):
    job(f"{PAIRS}_{i}", f"bash j_deriv.sh pairs {i} {NSHARD}", need=(S8,))
job("deriv_prof", "bash j_deriv.sh profiles", need=(S8F,))
# P2 labels: shards on reduced windows; pairs whose verification needs the full window are deferred and completed
# by label_redo; then merge + training sets (CPU)
for i in range(NSHARD):
    job(f"label_{i}", f"bash j_relabel.sh {i} {NSHARD}", after=[f"{PAIRS}_{k}" for k in range(NSHARD)], need=(S8,))
job("label_redo", f"bash j_relabel.sh redo {NSHARD}", after=[f"label_{k}" for k in range(NSHARD)] + ["deriv_prof"], need=(S8F,))
job("label", f"bash j_labelmerge.sh {NSHARD}", after=["label_redo"], need=())
# P3 students (QLoRA on a shared card), P4 evaluations, P5 programmatic benchmark -- main student first
for v in STUD:
    job(f"train_{v}", f"bash j_train.sh {v}", after=["label"], need=(ST,))
    job(f"eval_{ROLE[v]}", f"bash j_eval.sh {v} {ROLE[v]}", after=[f"train_{v}", "deriv_prof"], need=(S8, S4))
    if ROLE[v] != "nv":
        job(f"suff_{ROLE[v]}", f"bash j_suff.sh {v} {ROLE[v]}", after=[f"train_{v}"], need=(S8, S4))
for s in (1, 2):
    job(f"train_s{s}", f"bash j_train.sh tnet21_s{s} {s}", after=["label"], need=(ST,))
    job(f"eval_s{s}", f"bash j_eval.sh tnet21_s{s} s{s}", after=[f"train_s{s}", "deriv_prof"], need=(S8, S4))
job("eval_zs", "bash j_eval.sh none zs", after=["deriv_prof", "eval_main"], need=(S8,))
job("suff_ref", "bash j_suff.sh none ref", after=["suff_main"], need=(S8,))          # 8B reasoning reference row
CORE_EVAL = [f"eval_{ROLE[v]}" for v in STUD]
job("latency", "bash j_latency.sh", after=["eval_main", "suff_main"], need=(S8F, S4))
job("judge_qwen25-7b", "bash j_judge.sh qwen25-7b core", after=CORE_EVAL + ["eval_zs"], need=(SJ,))
job("judge_llama31", "bash j_judge.sh llama31 core", after=CORE_EVAL + ["eval_zs"], need=(S8,))
job("judge_seeds", "bash j_judge.sh qwen25-7b seeds", after=["eval_s1", "eval_s2", "judge_qwen25-7b"], need=(SJ,))
job("tables", "bash j_tables.sh", after=["judge_qwen25-7b", "judge_llama31", "judge_seeds", "latency",
                                        "suff_main", "suff_nd", "suff_w", "suff_nr"], need=())
PORTS = [18450 + 10 * i for i in range(40)]      # rotated: a port is not reused right after a job released it
                                                  # (its servers may still be shutting down and hold the socket)

# ------------------------------------------------------------------------------------------------------- state
STATE = f"{LOGS}/{NAME}.state.json"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f"{LOGS}/{NAME}.log", "a") as f:
        f.write(line + "\n")


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"jobs": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def gpu_free():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return {}
    free = {}
    for line in out.strip().splitlines():
        i, used, total = [int(x.strip()) for x in line.split(",")]
        if i in ALLOWED:
            free[i] = max(0, total - used - MARGIN)
    return free


def place(need, free, busy):
    """GPUs for the slots of a job, or None.  One slot: the freest GPU that fits.  Two slots: one GPU with room for
    both (same placement for every server) if available, else the freest fitting GPU per slot, distinct."""
    cands = sorted([g for g in free if g not in busy], key=lambda g: -free[g])
    if not need:
        return []
    if len(need) == 1:
        return [g for g in cands if free[g] >= need[0]][:1] or None
    both = [g for g in cands if free[g] >= sum(need) + 1000]
    if both:
        return [both[0]]
    g1 = next((g for g in cands if free[g] >= need[0]), None)
    if g1 is None:
        return None
    g2 = next((g for g in cands if g != g1 and free[g] >= need[1]), None)
    return [g1, g2] if g2 is not None else None


def alive(pid):
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as fp:
            return fp.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def done(jid):
    return os.path.exists(f"{LOGS}/J_{jid}.done")


def main():
    os.chdir(CODE)
    st = load_state()
    procs = {}
    for jid, s in st["jobs"].items():
        if s.get("status") == "running" and s.get("pid") and alive(s["pid"]):
            log(f"adopting running job {jid} pid={s['pid']}")
        elif s.get("status") == "running":
            s["status"] = "failed" if not done(jid) else "done"
            s["next"] = time.time() + 60
        elif s.get("status") == "done" and not done(jid):
            # the marker is the record of a finished job; one that was removed on purpose makes the job pending again
            s.update({"status": "pending", "attempts": 0})
            log(f"marker of {jid} is gone: the job is pending again")
    with open(f"{LOGS}/OWNER", "w") as f:
        f.write(f"{NAME} pid={os.getpid()} started {time.strftime('%Y-%m-%d %H:%M:%S')} gpus={ALLOWED}\n")
    log(f"scheduler start; allowed GPUs {ALLOWED}; {len(J)} jobs")
    noted, peak = time.time(), {}
    while True:
        if os.path.exists(f"{LOGS}/{UP}.stop"):
            for jid, pr in procs.items():
                try:
                    os.killpg(pr.pid, signal.SIGTERM)
                except Exception:
                    pass
            log("stop requested; exiting")
            return
        if os.path.exists(f"{LOGS}/{UP}.reload"):      # exit WITHOUT killing jobs: a new instance adopts them
            os.remove(f"{LOGS}/{UP}.reload")
            save_state(st)
            log("reload requested; exiting, running jobs continue")
            return
        # reap finished jobs
        for jid, pr in list(procs.items()):
            rc = pr.poll()
            if rc is None:
                continue
            s = st["jobs"][jid]
            if rc == 0 and done(jid):
                s["status"] = "done"
                log(f"DONE {jid} ({(time.time() - s['start']) / 60:.0f} min)")
            else:
                s["status"], s["attempts"] = "failed", s.get("attempts", 0) + 1
                # most failures are races for memory with other users' processes: retry soon, back off gently
                s["next"] = time.time() + min(900, 150 * s["attempts"])
                log(f"FAILED {jid} rc={rc} attempt={s['attempts']}; retry in {(s['next'] - time.time()) / 60:.0f} min; see logs/j_{jid}.log")
            del procs[jid]
            save_state(st)
        # adopted jobs from a previous scheduler instance
        for jid, s in st["jobs"].items():
            if s.get("status") == "running" and jid not in procs and s.get("pid"):
                if not alive(s["pid"]):
                    s["status"] = "done" if done(jid) else "failed"
                    s["next"] = time.time() + 60
                    log(f"adopted job {jid} ended: {s['status']}")
                    save_state(st)
        for j in J:
            if done(j["id"]) and st["jobs"].get(j["id"], {}).get("status") != "done":
                st["jobs"].setdefault(j["id"], {})["status"] = "done"
        if os.path.exists(f"{LOGS}/{UP}.pause"):
            time.sleep(POLL)
            continue
        free = gpu_free()
        for g, f in free.items():
            peak[g] = max(peak.get(g, 0), f)
        # a card of one of our own jobs is off limits while that job is still taking its memory: for SETTLE seconds
        # after the launch of a job with servers (a vLLM server allocates its whole share when it starts and keeps
        # it), and for as long as a training job runs (its footprint moves with the batch)
        busy = {g for jid, s in st["jobs"].items() if s.get("status") == "running" for g in s.get("gpus", [])
                if "_train_" in jid or time.time() - s.get("start", 0) < SETTLE}
        used_ports = {s.get("port") for s in st["jobs"].values() if s.get("status") == "running"}
        launched = False
        for j in J:
            s = st["jobs"].setdefault(j["id"], {"status": "pending", "attempts": 0})
            if s["status"] in ("done", "running") or done(j["id"]):
                continue
            if s["status"] == "failed" and time.time() < s.get("next", 0):
                continue
            if not all(done(d) for d in j["after"]):
                continue
            gpus = place(j["need"], free, busy)
            if gpus is None:
                continue
            k0 = st.get("port_idx", 0)
            port = next(PORTS[(k0 + i) % len(PORTS)] for i in range(len(PORTS)) if PORTS[(k0 + i) % len(PORTS)] not in used_ports)
            st["port_idx"] = (PORTS.index(port) + 1) % len(PORTS)
            f1 = free[gpus[0]] if gpus else 0
            f2 = free[gpus[1]] if len(gpus) > 1 else f1
            env = dict(os.environ, J_GPU=",".join(map(str, gpus)), J_PORT=str(port), J_JOB=j["id"],
                       J_FREE=str(f1), J_FREE2=str(f2), CM_ROOT=P)
            with open(f"{LOGS}/j_{j['id']}.log", "a") as lf:
                lf.write(f"\n===== launch {time.strftime('%m-%d %H:%M:%S')} gpus={gpus} port={port} free={f1}/{f2}\n")
                pr = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, stdout=lf, stderr=subprocess.STDOUT,
                                      cwd=CODE, start_new_session=True)
            procs[j["id"]] = pr
            s.update({"status": "running", "pid": pr.pid, "gpus": gpus, "port": port, "start": time.time()})
            busy |= set(gpus)
            used_ports.add(port)
            log(f"LAUNCH {j['id']} on GPUs {gpus} port {port} free={f1}/{f2} ({j['cmd']})")
            save_state(st)
            launched = True
            break          # one launch per poll: let the servers take their memory before measuring again
        if all(done(j["id"]) for j in J):
            log("ALL JOBS DONE")
            open(f"{LOGS}/{UP}.done", "w").close()
            return
        if time.time() - noted > NOTE:
            ready = [j["id"] for j in J if st["jobs"].get(j["id"], {}).get("status") not in ("done", "running")
                     and not done(j["id"]) and all(done(d) for d in j["after"])]
            if ready:
                log(f"waiting: {len(ready)} jobs ready ({', '.join(ready[:4])}); free MiB now {free}, "
                    f"largest seen since the last note {peak}")
            noted, peak = time.time(), {}
        time.sleep(45 if launched else POLL)


if __name__ == "__main__":
    os.makedirs(LOGS, exist_ok=True)
    main()
