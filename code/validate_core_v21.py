"""Verify that all reported v21 variants use the same frozen questions and complete QA labels."""
import json
import math
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def answers(doc):
    return sorted((a["scenario"], a["qkey"], a["question"], a["gold"]) for a in doc["answers"])


def validate(root):
    main = root / "main3"
    reference = {p.name: answers(read(p)) for p in (main / "comm20").glob("*.json")}
    if len(reference) != 201:
        raise ValueError("Expected 201 frozen histories")
    variants = ["comm21", "comm21_nd", "comm21_w", "comm21_nv", "comm21_nr"]
    extra = ["comm21_sel", "comm21_overwrite", "comm21_zs"]
    methods = variants + ["comm21_s1", "comm21_s2"] + extra
    for method in methods:
        files = {p.name: p for p in (main / method).glob("*.json")}
        if files.keys() != reference.keys():
            raise ValueError(f"Wrong set of histories for {method}: {len(files)}")
        for name, path in files.items():
            doc = read(path)
            if answers(doc) != reference[name] or doc["strategy"] != method:
                raise ValueError(f"Question or strategy mismatch in {path}")
            if any(a.get("label_qwen25-7b") not in (True, False) for a in doc["answers"]):
                raise ValueError(f"Missing binary judgment in {path}")
    qa = {r["method"]: r for r in read(root / "t40_main_qwen25-7b.json")}
    for m in methods:
        for key in ("corr", "corr_direct", "corr_derived", "both"):
            value, n = qa[m][key]
            if n != qa["comm20"][key][1] or n <= 0 or not math.isfinite(value):
                raise ValueError(f"Inconsistent {key} for {m}")
    direct = read(root / "xtab.json")
    for m in variants + extra:
        if m not in direct or direct[m]["n"] <= 0 or any(not math.isfinite(direct[m][k]) for k in ("sound", "stale")):
            raise ValueError(f"Incomplete direct evaluation for {m}")
    bench = read(root / "suffbench_v21.json")
    for m in ("copy", "ledger_all", "comm21", "comm21_nd", "comm21_w", "comm21_nr"):
        for group, ks in (("acc", ("all", "insufficient", "k_sum")), ("chain", ("1", "100"))):
            if any(not math.isfinite(bench[group][m][k]) for k in ks):
                raise ValueError(f"Incomplete programmatic evaluation for {m}")
    latency = read(root / "latency_v21.json")
    for m in ("comm21", "comm21_sel", "ncf", "coarse", "rebuild", "recompress"):
        if latency[m]["n"] != 30:
            raise ValueError(f"Incomplete latency sample for {m}")
    loc = {r["method"]: r for r in read(root / "t40_locomo_qwen25-7b.json")}
    for m in ("comm21", "comm21_nd", "comm21_nr", "ncf", "coarse", "rebuild"):
        if m not in loc or loc[m].get("n_corr", 0) <= 0:
            raise ValueError(f"Incomplete LoCoMo evaluation for {m}")
    print("Verified frozen QA cases, judgments, repair measures, chains, LoCoMo and latency sample sizes.")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
