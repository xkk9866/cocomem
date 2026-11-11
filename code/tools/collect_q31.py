"""Collect and render the manuscript when the current q31 queue finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import rr
from watch_q30 import LOCAL, REMOTE, WORKSPACE, extract_results


def collect():
    while True:
        try:
            code, state, err = rr.run(
                f"cd {REMOTE} && if test -f logs/Q31.done && test -s r31.tgz; "
                "then echo ready; elif test -f logs/Q31.failed; then echo failed; "
                "else echo running; fi", timeout=30)
        except OSError as exc:
            print(f"Connection interrupted: {exc}", flush=True)
            time.sleep(120)
            continue
        if code:
            raise RuntimeError(err)
        if state.strip() == "ready":
            break
        if state.strip() == "failed":
            raise RuntimeError("Q31.failed present; inspect the remote step log")
        time.sleep(120)

    archive = LOCAL / "r31.tgz"
    part = archive.with_suffix(".tgz.part")
    rr.get(f"{REMOTE}/r31.tgz", str(part))
    directory = LOCAL / "multiseed_v20"
    extract_results(part, directory)
    os.replace(part, archive)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert set(report["metrics"]) == {
        f"qa/{m}/corr" for m in json.loads((WORKSPACE / "code/multiseed_v20.json").read_text())["lme_methods"]}
    paper = WORKSPACE / "paper/icde"
    if "make_v21.py" in (paper / "main.tex").read_text(encoding="utf-8"):
        subprocess.run([sys.executable, str(paper / "make_multiseed_v20.py"),
                        "--summary", str(directory / "summary.json"), "--out", str(directory / "tables")],
                       cwd=WORKSPACE, check=True)
        print("Verified v20 results and tables collected alongside the separate v21 manuscript.", flush=True)
    else:
        subprocess.run([sys.executable, str(paper / "finalize_v20.py")], cwd=WORKSPACE, check=True)
        print("Results collected and manuscript rendered for visual review.", flush=True)


if __name__ == "__main__":
    collect()
