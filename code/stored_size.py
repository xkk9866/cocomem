import json, glob, os, sys
sys.path.insert(0, "/data/yaokang/corrmem/code")
from cocomem import ntok
ROOT = "/data/yaokang/corrmem"
E = json.load(open(f"{ROOT}/results/pub_export.json"))["main3"]
qids = sorted(E["amem"])
tot, txt, meta, ra = [], [], [], []
for q in qids:
    d = json.load(open(f"{ROOT}/states/amem/{q}.json"))
    t = sum(ntok(i["text"]) for i in d["items"])
    m = sum(ntok(i.get("context", "")) + ntok(", ".join(i.get("keywords", []))) + ntok(", ".join(i.get("tags", []))) for i in d["items"])
    txt.append(t); meta.append(m); tot.append(t + m)
    r = json.load(open(f"{ROOT}/states/readagent/{q}.json"))
    ra.append(sum(ntok(i["text"]) + ntok(i.get("gist", "")) for i in r["items"]))
n = len(qids)
print("n", n, "amem text", sum(txt)/n, "meta", sum(meta)/n, "total", sum(tot)/n, "| readagent pages+gists", sum(ra)/n)
