"""Collect the completed v21 manuscript bundle and rebuild the verified PDF."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import rr
from watch_q30 import extract_results

WS = Path(__file__).resolve().parents[2]
OUT = WS / "results_v5"
REMOTE = "/data/yaokang/corrmem"

def log(message):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUT / "collect40.log").open("a", encoding="utf-8") as stream:
        print(line, file=stream)

def collect():
    code, out, err = rr.run(f"cd {REMOTE} && if test -f logs/SCHED21.done && test -s r40.tgz; then sha256sum r40.tgz; else echo PENDING; fi", timeout=120)
    if code:
        raise RuntimeError(err[-500:])
    if out.strip() == "PENDING":
        return False
    expected = out.strip().split()[0]
    part = OUT / "r40.tgz.part"
    rr.get(f"{REMOTE}/r40.tgz", str(part))
    with part.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Downloaded bundle checksum mismatch")
    archive = OUT / "r40.tgz"
    part.replace(archive)
    count = extract_results(archive, OUT / "results")
    log(f"Validated and extracted {count} files; sha256={actual}")
    cmd = [sys.executable, str(WS / "paper/icde/finalize_v20.py"), "--version", "21"]
    marker = os.environ.get("COCOMEM_PDF_MARKER")
    if not marker:
        markers = sorted((Path.home() / ".codex/plugins/cache/openai-primary-runtime/pdf").glob("*/skills/pdf/container_tools/mark_artifact_operation_started.mjs"))
        if markers:
            marker = str(markers[-1])
    if marker:
        cmd += ["--marker", marker]
    subprocess.run(cmd, check=True)
    log("Compiled final v21 PDF and rendered pages for visual review")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()
    while True:
        try:
            if collect() or args.once:
                break
        except Exception as exc:
            log(f"Collection failed: {exc}")
            if args.once:
                raise
        time.sleep(args.interval)
