"""Turn results/table_*.json and lineage_quality.json into LaTeX tables for the paper."""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "/data/yaokang/corrmem/results"
OUT = sys.argv[2] if len(sys.argv) > 2 else R + "/tables.tex"

NAMES = {"rawrag": "RawRAG (uncompressed)", "membank": "MemoryBank", "mem0": "Mem0", "amem": "A-Mem",
         "append": "Append", "simupdate": "Sim-update (Mem0-style)", "self": "Self-lineage edit",
         "self_span": "Self-lineage span edit", "coarse": "Anchored unit edit", "ncf": "\textbf{CoCoMem (NCF)}",
         "recompress": "Anchored recompression", "rebuild": "Rebuild (reference)",
         "spec": "\\textbf{CoCoMem (Jacobi)}", "spec_ncf": "\\quad NCF closure",
         "spec_overwrite": "\\quad w/o version history", "oracle": "Oracle reader"}
GROUPS = [["rawrag", "membank", "mem0"], ["append", "simupdate", "self", "self_span", "coarse", "recompress",
                                           "rebuild"], ["ncf", "spec", "spec_ncf", "spec_overwrite"]]


def f(x):
    return f"{x:.1f}"


def main_table(rows):
    by = {r["method"]: r for r in rows}
    best = {}
    keys = ["orig", "corr", "corr_direct", "corr_derived", "fid", "hist", "both", "keep", "probe"]
    cand = [m for g in GROUPS[1:] for m in g if m in by and m not in ("rebuild",)]
    for k in keys:
        best[k] = max(by[m][k][0] for m in cand)
    L = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{LongMemEval-Corr test set ($n=%d$ questions; S1 metrics on %d oracle-valid corrections). Accuracy (\\%%), judged by Qwen2.5-7B. Fid.\\ = $P(\\text{S1 correct}\\mid\\text{S0 correct})$; Both = corrected and history answers both correct. Tokens = LLM input+output tokens per correction. $p$: exact McNemar test of Corr against CoCoMem. Bold: best compressed memory.}"
         % (by["ncf"]["n"], by["ncf"]["corr"][1]),
         "\\label{tab:main}",
         "\\begin{tabular}{lrrrrrrrrrrr}\\toprule",
         "Method & Orig & Corr & Direct & Derived & Fid. & Hist & Both & Keep & Probe & Tokens & $p$\\\\\\midrule"]
    for gi, g in enumerate(GROUPS):
        for m in g:
            if m not in by:
                continue
            r = by[m]
            cells = []
            for k in keys:
                v = r[k][0]
                s = f(v)
                if m in cand and abs(v - best[k]) < 1e-9:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            tok = "0" if r["c_tok"] == 0 else f"{r['c_tok'] / 1000:.1f}k"
            p = r.get("p_vs_ref", 1.0)
            ps = "ref." if m == "ncf" else (f"{p:.3f}" if p >= 0.001 else "$<$.001")
            L.append(f"{NAMES[m]} & " + " & ".join(cells) + f" & {tok} & {ps}\\\\")
        if gi < len(GROUPS) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(L)


def lineage_table(q):
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Predicting the commutator support (held-out users, %d (claim, unit) pairs, %d sentences; %.1f\\%% of sentences are in the support). Jacobi verification computes the support exactly.}"
         % (q["ncf"]["unit"]["n"], q["ncf"]["sent"]["n"], 100 * q["ncf"]["sent"]["pos"]),
         "\\label{tab:lineage}",
         "\\begin{tabular}{lrrrrr}\\toprule",
         "Estimator & Sent.\\ P & Sent.\\ R & Sent.\\ F1 & Sent.\\ AUC & Unit F1\\\\\\midrule"]
    for k, n in (("coarse", "Anchored inputs (all)"), ("similarity", "Embedding similarity"),
                 ("self_cite", "Compressor citations"), ("ncf", "NCF (learned)")):
        s, u = q[k]["sent"], q[k]["unit"]
        auc = "0.500" if k == "coarse" else f"{s['auc']:.3f}"
        if k == "self_cite":
            auc = f"{s['auc']:.3f}"
        L.append(f"{n} & {s['p']:.3f} & {s['r']:.3f} & {s['f1']:.3f} & {auc} & {u['f1']:.3f}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    by = q.get("by_level", {})
    L2 = ["\\begin{table}[t]\\centering\\small",
          "\\caption{Sentence-level recall of the commutator support by unit level (held-out users).}",
          "\\label{tab:level}", "\\begin{tabular}{lrr}\\toprule", "Level & Citations & NCF\\\\\\midrule"]
    for lev in ("note", "digest", "profile"):
        if lev in by:
            L2.append(f"{lev.capitalize()} & {by[lev]['self_cite']['sent']['r']:.3f} & {by[lev]['ncf']['sent']['r']:.3f}\\\\")
    L2 += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L) + "\n\n" + "\n".join(L2)


def locomo_table(rows):
    keys = ["orig", "corr", "hist", "keep", "both"]
    order = ["rawrag", "membank", "mem0", "amem", "append", "simupdate", "self", "coarse", "recompress", "rebuild", "ncf", "spec"]
    NAMES["spec"] = "\quad Jacobi verification"
    by = {r["method"]: r for r in rows}
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{LoCoMo-Corr: %d sequential corrections over 10 conversations, 351 unaffected questions. Accuracy (\\%%); tokens per correction.}" % by["spec"]["n_corr"],
         "\\label{tab:locomo}", "\\begin{tabular}{lrrrrrr}\\toprule",
         "Method & Orig & Corr & Hist & Keep & Both & Tokens\\\\\\midrule"]
    for m in order:
        if m not in by:
            continue
        r = by[m]
        L.append(f"{NAMES[m]} & " + " & ".join(f(r[k]) for k in keys) + f" & {r['tok'] / 1000:.1f}k\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    parts = []
    if os.path.exists(R + "/table_main.json"):
        parts.append(main_table(json.load(open(R + "/table_main.json"))))
    if os.path.exists(R + "/lineage_quality.json"):
        parts.append(lineage_table(json.load(open(R + "/lineage_quality.json"))))
    if os.path.exists(R + "/table_locomo.json"):
        parts.append(locomo_table(json.load(open(R + "/table_locomo.json"))))
    open(OUT, "w").write("\n\n".join(parts) + "\n")
    print(open(OUT).read())
