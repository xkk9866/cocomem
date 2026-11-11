"""Control memories for the write-path comparison: the v21 memories (prose first, records attributed by a second
pass) without their profile unit, i.e. with the two derived levels of the record-first memories.  Claims, notes and
digests are copied unchanged; the attributed records are read from the v21 record files.

  CM_RUN_ROOT=/data/yaokang/corrmem/v21np python strip_profile.py"""
import glob
import json
import os

SRC = "/data/yaokang/corrmem"
ROOT = os.environ.get("CM_RUN_ROOT", f"{SRC}/v21np")
assert os.path.abspath(ROOT) != SRC

from splits import add_test200, make_splits          # noqa: E402

os.makedirs(f"{ROOT}/states", exist_ok=True)
for name, target in (("deriv", f"{SRC}/states/deriv"), ("deriv_locomo", f"{SRC}/states/deriv_locomo")):
    if not os.path.lexists(f"{ROOT}/states/{name}"):
        os.symlink(target, f"{ROOT}/states/{name}")
for name, target in (("results", f"{SRC}/v22/results"), ("models", f"{SRC}/models"), ("data", f"{SRC}/data")):
    if not os.path.lexists(f"{ROOT}/{name}"):
        os.symlink(target, f"{ROOT}/{name}")
jobs = [("cocomem", sorted(add_test200(make_splits())["test200"])),
        ("locomo_cocomem", sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{SRC}/states/locomo_cocomem/*.json")))]
for kind, ids in jobs:
    os.makedirs(f"{ROOT}/states/{kind}", exist_ok=True)
    n = k = 0
    for q in ids:
        src, dst = f"{SRC}/states/{kind}/{q}.json", f"{ROOT}/states/{kind}/{q}.json"
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        d = json.load(open(src))
        gone = [x for x, u in d["units"].items() if u["level"] == "profile"]
        for x in gone:
            del d["units"][x]
        json.dump(d, open(dst + ".tmp", "w"))
        os.replace(dst + ".tmp", dst)
        n += 1
        k += len(gone)
    print(f"{kind}: {n} memories written, {k} profile units left out; total {len(glob.glob(f'{ROOT}/states/{kind}/*.json'))}", flush=True)
