"""v22: where a corrected answer is decided (CPU, after the export).  For every CoCoMem variant with result files:
  - the valid evidence corrections split by what the correction path did: the intake revised no slot ("noslot"), a
    slot was revised and no stored sentence changed ("norep"), some sentence of the closure was rewritten ("rep");
    accuracy of the corrected question in each class, direct and derived questions separately;
  - the intake over all evidence / unrelated corrections (share with a revised slot, slots per correction);
  - the size of a repair (units and sentences of the closure, sentences mapped and rewritten, tokens, seconds);
  - record writes, and how many of the operator's raw output records altered the entry of an operand that the
    correction did not address (what keyed maintenance leaves out of the stored record).
  python err_quant22.py comm22,comm22_blind,... [out.json]"""
import glob
import json
import os
import sys
from collections import Counter

P = "/data/yaokang/corrmem"
RC = os.environ.get("CM_RUN_ROOT", f"{P}/v22") + "/results"
DIRECT = ("single-session-user", "single-session-assistant", "knowledge-update")


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def main():
    methods = sys.argv[1].split(",")
    E = json.load(open(f"{RC}/pub_export_qwen25-7b.json"))["main3"]
    v1 = {q for q, r in E["oracle"].items() if r["lab"].get("S1:q") is True}
    out = {}
    for m in methods:
        files = sorted(glob.glob(f"{RC}/main3/{m}/*.json"))
        if not files or m not in E:
            continue
        C, S = Counter(), Counter()
        qids = {"noslot": [], "norep": []}
        for f in files:
            d = json.load(open(f))
            q = d["qid"]
            for sc in ("S1", "S2"):
                rep = d.get("repair", {}).get(sc)
                if rep is None:
                    continue
                S[sc, "n"] += 1
                S[sc, "slot"] += bool(rep.get("revised"))
                S[sc, "slots"] += len(rep.get("revised") or [])
                S[sc, "units"] += len(rep.get("closure") or [])
                S[sc, "units_rewritten"] += len(rep.get("repaired") or [])
                S[sc, "sents"] += rep.get("closure_sents") or 0
                S[sc, "mapped"] += rep.get("mapped_sents") or 0
                S[sc, "rewritten"] += sum((rep.get("rewritten_sents") or {}).values())
                S[sc, "rec_writes"] += rep.get("rec_writes") or 0
                S[sc, "rec_other"] += rep.get("rec_other_operand_altered") or 0
                S[sc, "tok8"] += sum((rep.get("usage") or {}).get(k, 0) for k in ("in", "out"))
                S[sc, "tok4"] += sum((rep.get("t_usage") or {}).get(k, 0) for k in ("in", "out"))
                S[sc, "sec"] += rep.get("time") or 0
            if q not in v1 or q not in E[m] or "S1:q" not in E[m][q]["lab"]:
                continue
            rep = d["repair"]["S1"]
            lab = bool(E[m][q]["lab"]["S1:q"])
            kind = "dir" if d["qtype"] in DIRECT else "der"
            cls = "noslot" if not rep.get("revised") else ("rep" if rep.get("repaired") else "norep")
            for k in (kind, "all"):
                C[k, cls, "n"] += 1
                C[k, cls, "ok"] += lab
                C[k, "any", "n"] += 1
                C[k, "any", "ok"] += lab
            if cls in qids:
                qids[cls].append([q, kind, lab])
            a = [x for x in d["answers"] if x["scenario"] == "S1" and x["qkey"] == "q"]
            if a and a[0].get("stale_in_ctx") is not None:
                C["stale_ctx", a[0]["stale_in_ctx"], "n"] += 1
                C["stale_ctx", a[0]["stale_in_ctx"], "ok"] += lab
        r = {"files": len(files)}
        for kind in ("all", "dir", "der"):
            for cls in ("any", "noslot", "norep", "rep"):
                n = C[kind, cls, "n"]
                r[f"{kind}.{cls}.n"] = n
                r[f"{kind}.{cls}.acc"] = pct(C[kind, cls, "ok"], n)
        for v in (True, False):
            r[f"stale_ctx.{int(v)}.n"] = C["stale_ctx", v, "n"]
            r[f"stale_ctx.{int(v)}.acc"] = pct(C["stale_ctx", v, "ok"], C["stale_ctx", v, "n"])
        for sc in ("S1", "S2"):
            n = S[sc, "n"]
            if not n:
                continue
            r[f"{sc}.n"] = n
            r[f"{sc}.slot"] = pct(S[sc, "slot"], n)
            for k in ("slots", "units", "units_rewritten", "sents", "mapped", "rewritten", "tok8", "tok4", "sec"):
                r[f"{sc}.{k}"] = round(S[sc, k] / n, 2)
            r[f"{sc}.rec_writes"] = S[sc, "rec_writes"]
            r[f"{sc}.rec_other"] = S[sc, "rec_other"]
            r[f"{sc}.rec_other_pct"] = pct(S[sc, "rec_other"], S[sc, "rec_writes"])
        r["qids"] = qids
        out[m] = r
        print(f"== {m} files={len(files)}")
        for kind in ("all", "dir", "der"):
            print("  " + kind + ": " + "  ".join(f"{cls} n={r[f'{kind}.{cls}.n']} acc={r[f'{kind}.{cls}.acc']}" for cls in ("any", "noslot", "norep", "rep")))
        for sc in ("S1", "S2"):
            if f"{sc}.n" in r:
                print(f"  {sc}: " + " ".join(f"{k}={r[f'{sc}.{k}']}" for k in ("n", "slot", "slots", "units", "units_rewritten", "sents", "mapped",
                                                                         "rewritten", "tok8", "tok4", "sec", "rec_writes", "rec_other", "rec_other_pct")))
        print("  stale value in the reader's context:", {v: (r[f"stale_ctx.{v}.n"], r[f"stale_ctx.{v}.acc"]) for v in (1, 0)})
        print("  noslot:", qids["noslot"])
        print("  norep:", qids["norep"])
    dst = sys.argv[2] if len(sys.argv) > 2 else f"{RC}/err_quant_v22.json"
    json.dump(out, open(dst, "w"), indent=1)


if __name__ == "__main__":
    main()
