import json, os
R = "/data/yaokang/corrmem/results/main3"
E = json.load(open("/data/yaokang/corrmem/results/pub_export.json"))["main3"]
DIRECT = ("single-session-user", "single-session-assistant", "knowledge-update")
v1 = sorted(q for q, r in E["oracle"].items() if r["lab"].get("S1:q") is True and q in E["comm21"])
from collections import Counter
C = Counter()
rows = []
for q in v1:
    d = json.load(open(f"{R}/comm21/{q}.json"))
    rep = d["repair"]["S1"]
    lab = E["comm21"][q]["lab"].get("S1:q")
    kind = "direct" if d["qtype"] in DIRECT else "derived"
    a = [x for x in d["answers"] if x["scenario"] == "S1" and x["qkey"] == "q"][0]
    nrev = len(rep.get("revised", []))
    ncl = len(rep.get("closure", []))
    nrep = len(rep.get("repaired", []))
    cls = "nogrnd" if nrev == 0 else ("allrep" if nrep == ncl else ("partrep" if nrep else "norep"))
    C[(kind, cls, lab)] += 1
    rows.append((q, kind, cls, lab, a.get("stale_in_ctx"), ncl, nrep, rep.get("closure_sents"), rep.get("mapped_sents"), sum((rep.get("rewritten_sents") or {}).values())))
for kind in ("direct", "derived"):
    print("==", kind)
    for cls in ("nogrnd", "norep", "partrep", "allrep"):
        t = C[(kind, cls, True)]; f = C[(kind, cls, False)]
        print(f"  {cls:8s} n={t+f:3d} acc={100*t/max(1,t+f):5.1f}  (T={t}, F={f})")
print("== stale_in_ctx by label (all 126)")
S = Counter((r[3], r[4]) for r in rows); print(S)
print("== closure/repaired sizes: mean closure units", sum(r[5] for r in rows)/len(rows), "mean repaired", sum(r[6] for r in rows)/len(rows))
print("== sentences: closure", sum(r[7] or 0 for r in rows)/len(rows), "mapped", sum(r[8] or 0 for r in rows)/len(rows), "rewritten", sum(r[9] for r in rows)/len(rows))
# the no-grounding cases
print("== no grounding qids:", [r[0] for r in rows if r[2] == "nogrnd"])
# also what does membank/rawrag do on nogrnd
for q in [r[0] for r in rows if r[2] == "nogrnd"]:
    print(q, "mb", E["membank"][q]["lab"].get("S1:q"), "rr", E["rawrag"][q]["lab"].get("S1:q"), "cm", E["comm21"][q]["lab"].get("S1:q"), E["comm21"][q]["qtype"])
