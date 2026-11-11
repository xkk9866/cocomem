"""Background poller: appends the server-side status (st40.sh) to results_v5/poll40.log every N seconds."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(WS, "results_v5", "poll40.log")
interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
while True:
    try:
        code, out, err = rr.run("bash /data/yaokang/corrmem/code/st40.sh 2>&1", timeout=120)
    except Exception as e:
        out = f"poll error: {e}\n"
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n" + out)
    time.sleep(interval)
