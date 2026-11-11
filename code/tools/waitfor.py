"""Poll the server until a shell condition succeeds (exit 0), then print status and exit."""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
import rr
cond, status = sys.argv[1], sys.argv[2]
interval = int(sys.argv[3]) if len(sys.argv) > 3 else 120
t0 = time.time()
while True:
    try:
        code, out, err = rr.run(f"({cond}) >/dev/null 2>&1 && echo READY; {status}", timeout=120)
        if "READY" in out:
            print(f"[after {time.time()-t0:.0f}s]\n" + out)
            break
    except Exception as e:
        pass
    time.sleep(interval)
