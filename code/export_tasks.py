"""Export the correction tasks as plain JSON so that the OFFICIAL baseline code (separate environments) can run them.
lme.json    : per question {qid, qtype, qdate, question, sessions, corr:{S1: session, S2: session}, queries:[[sc,key,q,gold,jtype]]}
locomo.json : per conversation {cid, persons, sessions, corr:[sessions...], queries:[[sc,key,q,gold,jtype,cat]]}"""
import json
import os

from data import load_lme
from locomo import corr_sessions, load_locomo, query_list
from scenarios import corr_session, load_corr, queries
from splits import add_test200, make_splits

OUT = "/data/yaokang/corrmem/bl/tasks"
os.makedirs(OUT, exist_ok=True)
sp = add_test200(make_splits())
items = {x["qid"]: x for x in load_lme()}
corr = load_corr()
rows = []
for split in ("test200", "dev"):
    for q in sp[split]:
        if q not in corr or q not in items:
            continue
        inst, rec = items[q], corr[q]
        c = {}
        for kind, sc in (("affect", "S1"), ("unrel", "S2")):
            if kind in rec:
                c[sc] = corr_session(inst, rec, kind)
        rows.append({"qid": q, "split": split, "qtype": inst["qtype"], "qdate": inst["qdate"], "question": inst["question"],
                     "sessions": inst["sessions"], "corr": c, "queries": queries(inst, rec),
                     "history_answer": rec.get("affect", {}).get("history_answer", "")})
json.dump(rows, open(f"{OUT}/lme.json", "w"))
print("lme", len(rows), sum(r["split"] == "test200" for r in rows))
corrl = json.load(open("/data/yaokang/corrmem/data/locomo_corr.json"))
rows = []
for cv in load_locomo():
    rec = corrl[cv["cid"]]
    rows.append({"cid": cv["cid"], "persons": cv["persons"], "sessions": cv["sessions"], "corr": corr_sessions(cv, rec),
                 "queries": query_list(rec), "n_aff": len(rec["affect"])})
json.dump(rows, open(f"{OUT}/locomo.json", "w"))
print("locomo", len(rows))
