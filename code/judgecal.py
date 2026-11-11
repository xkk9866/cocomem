"""Judge calibration against COMPUTED truth (no human labels are available, and the judges may not exceed 8B int4).
From the programmatic benchmark we build, for every sentence whose value changes under the correction, three texts whose
labels are known by construction:
  fresh   the sentence regenerated from the corrected values   -> stale: no , states the corrected information: yes
  stale   the stored sentence, untouched                        -> stale: yes, states the corrected information: no
  omitted the aggregate dropped (a sentence on another subject) -> stale: no , states the corrected information: no
The same three prompts that judge the repaired memories (comm_eval2: STALE, HAS, FACT) are run on them; we report the
accuracy of each question and of the conjunctions used in the paper (stale, missing, commute) against the truth.
Usage: python judgecal.py --model llama31 --port 18431  (one judge per call)"""
import argparse
import asyncio
import json

from comm_eval2 import STALE, HAS, FACT
from llm import LLM, gather_limited
from suffbench import build, sentences, truth, ROOT

yes = lambda r: r.strip().lower().startswith("yes")


async def main(a):
    llm = LLM(a.model, a.port, concurrency=96)
    cases, units = build(a.units, a.seed)
    rows = []
    for c in cases:
        if c["label"] == "copy":
            continue
        u = units[c["u"]]
        w = list(u["vals"])
        i = c["i"]
        newv = int(__import__("re").findall(r"\d+", c["new"])[0])
        w[i] = newv
        fresh = sentences(u["verb"], u["unit"], u["days"], w, u["T"])[c["kind"]]
        rows.append({"kind": c["kind"], "old": c["old"], "corr": c["new"],
                     "texts": {"fresh": fresh, "stale": c["sent"], "omitted": u["S"]["other"]}})

    async def one(r):
        out = {"kind": r["kind"]}
        for name, t in r["texts"].items():
            st, hs, ft = await asyncio.gather(*(llm.chat(P.format(old=r["old"], corr=r["corr"], text=t), max_tokens=3) for P in (STALE, HAS, FACT)))
            out[name] = {"stale": yes(st), "has": yes(hs), "fact": yes(ft)}
        return out

    R = await gather_limited([one(r) for r in rows], desc="cal", every=200)
    truth_ = {"fresh": {"stale": False, "has": True}, "stale": {"stale": True, "has": False}, "omitted": {"stale": False, "has": False}}
    res = {"n": len(R)}
    for q in ("stale", "has"):
        res[q] = {v: 100 * sum(x[v][q] == truth_[v][q] for x in R) / len(R) for v in truth_}
        res[q]["all"] = sum(res[q][v] for v in truth_) / 3
    # "missing" needs a reference that states the corrected information: here the fresh sentence is the reference
    miss = {v: 100 * sum((x["fresh"]["has"] and not x[v]["has"]) == (v != "fresh") for x in R if x["fresh"]["has"]) /
            max(1, sum(x["fresh"]["has"] for x in R)) for v in truth_}
    res["missing"] = miss | {"all": sum(miss.values()) / 3}
    com = {v: 100 * sum(((not x[v]["stale"]) and not (x["fresh"]["has"] and not x[v]["has"])) == (v == "fresh") for x in R if x["fresh"]["has"]) /
           max(1, sum(x["fresh"]["has"] for x in R)) for v in truth_}
    res["commute"] = com | {"all": sum(com.values()) / 3}
    res["fact_on_fresh"] = 100 * sum(x["fresh"]["fact"] for x in R) / len(R)
    res["fact_on_stale"] = 100 * sum(x["stale"]["fact"] for x in R) / len(R)
    res["fact_on_omitted"] = 100 * sum(x["omitted"]["fact"] for x in R) / len(R)
    json.dump({"res": res, "raw": R}, open(f"{ROOT}/results/judgecal_{a.model}.json", "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18431)
    ap.add_argument("--units", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    asyncio.run(main(ap.parse_args()))
