"""Fixed dev / train / test splits over the 440 eligible LongMemEval-S questions."""
import json
import os
import random

from data import load_lme

SPLIT_FILE = "/data/yaokang/corrmem/data/lme_splits.json"


def make_splits(n_dev=40, n_train=100, seed=13):
    if os.path.exists(SPLIT_FILE):
        return json.load(open(SPLIT_FILE))
    items = load_lme()
    rng = random.Random(seed)
    by = {}
    for x in items:
        by.setdefault(x["qtype"], []).append(x["qid"])
    dev, train, test = [], [], []
    N = len(items)
    for t, q in sorted(by.items()):
        q = sorted(q)
        rng.shuffle(q)
        kd = round(n_dev * len(q) / N)
        kt = round(n_train * len(q) / N)
        dev += q[:kd]
        train += q[kd:kd + kt]
        test += q[kd + kt:]
    sp = {"dev": sorted(dev), "train": sorted(train), "test": sorted(test)}
    json.dump(sp, open(SPLIT_FILE, "w"), indent=1)
    return sp


def add_test200(sp, n=200, seed=29):
    """Stratified 200-question test subset (main evaluation set)."""
    if "test200" in sp:
        t = sorted(sp["test200"])
        sp["test200"] = t[::2] + t[1::2]   # processing order: an interleaved half first (same subset for all methods)
        return sp
    items = {x["qid"]: x for x in load_lme()}
    rng = random.Random(seed)
    by = {}
    for q in sp["test"]:
        by.setdefault(items[q]["qtype"], []).append(q)
    out = []
    tot = len(sp["test"])
    for t, q in sorted(by.items()):
        q = sorted(q)
        rng.shuffle(q)
        out += q[:round(n * len(q) / tot)]
    sp["test200"] = sorted(out)
    json.dump(sp, open(SPLIT_FILE, "w"), indent=1)
    return sp


if __name__ == "__main__":
    s = add_test200(make_splits())
    print({k: len(v) for k, v in s.items()})
