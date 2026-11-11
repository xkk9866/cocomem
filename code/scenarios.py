"""Correction scenarios built from LongMemEval-Corr records."""
import datetime as dt
import json

CORR_PATH = "/data/yaokang/corrmem/data/lme_corr.json"


def parse_date(s):
    return dt.datetime.strptime(s[:10] + " " + s[-5:], "%Y/%m/%d %H:%M")


def fmt_date(d):
    return d.strftime("%Y/%m/%d (%a) %H:%M")


def corr_date(inst):
    last = max(parse_date(s["date"]) for s in inst["sessions"])
    q = parse_date(inst["qdate"])
    d = last + dt.timedelta(minutes=30)
    if d >= q:
        d = last + (q - last) / 2
    return fmt_date(d)


def corr_session(inst, rec, kind):
    js = rec[kind]
    return {"sid": f"corr_{kind}", "date": corr_date(inst), "evidence": False,
            "turns": [{"role": "user", "content": js["correction_message"]},
                      {"role": "assistant", "content": js.get("assistant_reply") or "Thanks for letting me know, I've updated that."}]}


def load_corr():
    return {r["qid"]: r for r in json.load(open(CORR_PATH))}


def queries(inst, rec):
    """List of (scenario, query_id, question, gold, judge_type)."""
    qs = [("S0", "q", inst["question"], inst["answer"], inst["qtype"])]
    if "affect" in rec:
        a = rec["affect"]
        jt = "temporal-corrected" if inst["qtype"] == "temporal-reasoning" else "knowledge-update"
        qs.append(("S1", "q", inst["question"], a["new_answer"], jt))
        qs.append(("S1", "hist", a["history_question"], a["history_answer"], "history"))
    if "unrel" in rec:
        u = rec["unrel"]
        qs.append(("S2", "q", inst["question"], inst["answer"], inst["qtype"]))
        qs.append(("S2", "probe", u["probe_question"], u["probe_answer"], "knowledge-update"))
    return qs
