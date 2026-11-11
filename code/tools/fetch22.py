"""Fetch r22.tgz (j22_tables.sh), unpack it into results_v22/ and rebuild the paper tables and numbers22.tex.
  python code/tools/fetch22.py [--no-fetch]"""
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DST = os.path.join(ROOT, "results_v22")
PAPER = os.path.join(ROOT, "paper", "icde")
sys.path.insert(0, HERE)

if "--no-fetch" not in sys.argv:
    import rr
    os.makedirs(DST, exist_ok=True)
    rr.get("/data/yaokang/corrmem/r22.tgz", os.path.join(DST, "r22.tgz"))
    with tarfile.open(os.path.join(DST, "r22.tgz")) as t:
        t.extractall(DST)
    print("fetched:", len(os.listdir(DST)), "files")


def arg(f):
    p = os.path.join(DST, f)
    return p if os.path.exists(p) and os.path.getsize(p) > 2 else "-"


ws = arg("write_stats_v22.json")
if ws == "-":
    ws = arg("write_stats_v22_partial.json")
cmd = [sys.executable, "make_v22.py", arg("pub_export_qwen25-7b.json"), os.path.join(DST, "out22.json"), PAPER,
       arg("pub_export_llama31.json"), ws, arg("suffbench_v22.json"), arg("latency_v22.json"), arg("xtab.json"),
       arg("err_quant_v22.json")]
print(" ".join(os.path.basename(c) for c in cmd))
r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
open(os.path.join(DST, "make_v22.log"), "w", encoding="utf-8").write(r.stdout + "\n[stderr]\n" + r.stderr)
print(r.stdout[-6000:])
print(r.stderr[-3000:])
