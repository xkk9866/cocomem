"""Watch the already-running q30 queue; fetch and finalize only on success.

No remote start, restart or termination is performed. --once is read-only.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import time

import rr


WORKSPACE = Path(__file__).resolve().parents[2]
LOCAL = WORKSPACE / "results_v5"
REMOTE = "/data/yaokang/corrmem"
TZ = timezone(timedelta(hours=8))

# Exact argument matching excludes this read-only Python probe itself.
PROBE = r'''
import json, os
from pathlib import Path
p = Path("/data/yaokang/corrmem")
def tail(path, n=8):
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 16000))
        return f.read().decode("utf-8", "replace").splitlines()[-n:]
def count(path):
    return sum(1 for x in path.glob("*.json")) if path.is_dir() else 0
log = p / "logs"
failure = log / "Q30.failed"
archive = p / "r30.tgz"
pids = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / "cmdline").read_bytes().split(b"\0")
        if any(Path(a.decode("utf-8", "replace")).name in {"q30.sh", "q30_recover.sh"} for a in args if a):
            pids.append(int(proc.name))
    except (OSError, ValueError):
        pass
out = {
    "done": (log / "Q30.done").exists(),
    "failed": failure.exists(),
    "failure": failure.read_text(errors="replace").strip() if failure.exists() else "",
    "queue_pids": sorted(pids),
    "q30_tail": tail(log / "q30.log", 14),
    "progress": {name: tail(log / name, 2) for name in
        ("q30_witness.log", "q30_tnet.log", "q30_train_tnet20.log", "q30_train_tnet20nw.log",
         "q30_train_tnet20nv.log", "q30_train_tnet20nr.log") if (log / name).exists()},
    "counts": {name: count(p / name) for name in
        ("states/witness", "states/witness_locomo", "results/main3/comm20",
         "results/main3/comm20_nw", "results/main3/comm20_nv", "results/main3/comm20_nr",
         "results/locomo/comm20", "results/locomo_reg/comm20")},
    "trained": [name for name in ("tnet20", "tnet20nw", "tnet20nv", "tnet20nr")
        if (p / "models" / name / "merged/config.json").exists()
        and (p / "models" / name / "train_meta.json").exists()],
    "archive_bytes": archive.stat().st_size if archive.exists() else 0,
}
print(json.dumps(out))
'''


def read_remote():
    code, out, err = rr.run("python3 -c " + shlex.quote(PROBE), timeout=90)
    if code:
        raise RuntimeError(f"Remote status probe failed (exit {code}): {err[-1000:]}")
    return json.loads(out)


def remote_state(snapshot):
    if snapshot["failed"]:
        return "failed", snapshot["failure"] or "Q30.failed exists"
    if snapshot["done"]:
        if not any("ALL DONE" in line for line in snapshot["q30_tail"]):
            return "failed", "Q30.done exists without ALL DONE in the final log"
        if not snapshot["archive_bytes"]:
            return "failed", "Queue marked done but r30.tgz is missing or empty"
        return "ready", "Remote experiments finished"
    if not snapshot["queue_pids"]:
        return "failed", "q30.sh is absent and neither success nor failure marker is present"
    return "running", "Remote queue is active"


def extract_results(archive, destination):
    """Validate every path first, then atomically write regular files only."""
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        checked, seen = [], set()
        for member in members:
            posix = PurePosixPath(member.name)
            if posix.is_absolute() or ".." in posix.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            target = destination.joinpath(*posix.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes results directory: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Unsupported archive member: {member.name}")
            if target in seen:
                raise ValueError(f"Duplicate archive path: {member.name}")
            seen.add(target)
            checked.append((member, target))
        written = 0
        for member, target in checked:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".q30-download")
            with tf.extractfile(member) as source, temporary.open("wb") as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            os.replace(temporary, target)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print one read-only status snapshot and exit")
    parser.add_argument("--interval", type=float, default=60, help="Polling interval in seconds (default: 60)")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("--interval must be at least 10 seconds")
    if args.once:
        try:
            snapshot = read_remote()
            state, detail = remote_state(snapshot)
            print(json.dumps({"state": state, "detail": detail, "remote": snapshot}, indent=2, ensure_ascii=False))
            return 2 if state == "failed" else 0
        except Exception as exc:
            print(json.dumps({"state": "connection_error", "detail": str(exc)}, ensure_ascii=False))
            return 3

    LOCAL.mkdir(parents=True, exist_ok=True)
    status_path = LOCAL / "q30_status.json"
    log_path = LOCAL / "q30_watch.log"
    last_fingerprint = None

    def publish(state, detail, snapshot=None, **extra):
        nonlocal last_fingerprint
        record = {"state": state, "detail": detail, "remote": snapshot, **extra}
        fingerprint = json.dumps(record, sort_keys=True, ensure_ascii=False)
        record["observed_at"] = datetime.now(TZ).isoformat(timespec="seconds")
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, status_path)
        if fingerprint != last_fingerprint:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{record['observed_at']} {state}: {detail}", flush=True)
            last_fingerprint = fingerprint

    while True:
        started = time.monotonic()
        try:
            snapshot = read_remote()
            state, detail = remote_state(snapshot)
        except Exception as exc:
            publish("connection_error", str(exc))
            return 3
        publish(state, detail, snapshot)
        if state == "failed":
            return 2
        if state == "ready":
            try:
                archive = LOCAL / "r30.tgz"
                partial = LOCAL / "r30.tgz.part"
                publish("downloading", "Downloading r30.tgz", snapshot)
                rr.get(f"{REMOTE}/r30.tgz", str(partial))
                if partial.stat().st_size != snapshot["archive_bytes"]:
                    raise RuntimeError("Downloaded archive size differs from remote size")
                count = extract_results(partial, LOCAL / "results")
                os.replace(partial, archive)
                script = WORKSPACE / "paper/icde/finalize_v20.py"
                if not script.is_file():
                    raise FileNotFoundError(f"Paper finalizer is missing: {script}")
                publish("finalizing", f"Extracted {count} files; generating tables and compiling paper", snapshot)
                output = LOCAL / "q30_finalize_stdout.log"
                with output.open("w", encoding="utf-8") as log:
                    result = subprocess.run([sys.executable, str(script)], cwd=script.parent,
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    publish("finalize_failed", f"Paper finalizer exited {result.returncode}; inspect {output}", snapshot)
                    return 4
                publish("completed", "Experiments downloaded and paper finalized", snapshot,
                        archive=str(archive), paper=str(script.parent / "main.pdf"))
                return 0
            except Exception as exc:
                publish("finalize_failed", str(exc), snapshot)
                return 4
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
