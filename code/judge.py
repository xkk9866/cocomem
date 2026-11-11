"""LLM-as-judge with the official LongMemEval answer-check templates (Qwen3-14B-AWQ judge)."""
import argparse
import asyncio
import glob
import json
import os

from llm import LLM, gather_limited

BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
        "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
        "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
        "only contains a subset of the information required by the answer, answer no. ")
TEMP = ("In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number "
        "of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
        "is 18), the model's response is still correct. ")
KU = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
      "contains the correct answer. Otherwise, answer no. If the response contains some previous information along "
      "with an updated answer, the response should be considered as correct as long as the updated answer is the "
      "required answer.")
TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."


def prompt(jtype, q, gold, hyp):
    if jtype in ("single-session-user", "single-session-assistant", "multi-session", "history"):
        t = BASE
    elif jtype == "temporal-reasoning":
        t = BASE + TEMP
    elif jtype == "temporal-corrected":
        t = KU + " " + TEMP
    else:  # knowledge-update, corrected answers, probes
        t = KU
    return (t + TAIL).format(q, gold, hyp)


async def judge_file(llm, path, key, primary=True):
    """Labels of one answer file under `key`.  The primary judge also sets the default `label`; a secondary judge
    (agreement study) writes only its own key."""
    d = json.load(open(path))
    changed = False
    for a in d["answers"]:
        if key in a and ("label" in a or not primary):
            continue
        r = await llm.chat(prompt(a["jtype"], a["question"], a["gold"], a["hyp"]), max_tokens=10, think=False)
        a[key] = "yes" in r.lower()
        if primary:
            a["label"] = a[key]
        changed = True
    if changed:      # merge only this judge's labels under a file lock: other judges write the file concurrently
        import fcntl
        with open(path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            fresh = json.load(open(path))
            for a, b in zip(fresh["answers"], d["answers"]):
                if key in b:
                    a[key] = b[key]
                    if primary:
                        a["label"] = b[key]
            tmp = f"{path}.{os.getpid()}.tmp"
            json.dump(fresh, open(tmp, "w"))
            os.replace(tmp, path)


async def main(args):
    llm = LLM(args.model, args.port, concurrency=192)
    files = []
    for pat in args.glob.split(","):
        files += sorted(glob.glob(pat))
    await gather_limited([judge_file(llm, f, "label_" + args.model, primary=not args.secondary) for f in files],
                         desc="judge", every=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/data/yaokang/corrmem/results/*/*/*.json")
    ap.add_argument("--port", type=int, default=18431)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--secondary", action="store_true", help="agreement study: write label_<model> only, keep `label`")
    asyncio.run(main(ap.parse_args()))
