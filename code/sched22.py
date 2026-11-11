"""v22 job graph (record-first compression) on the GPU-aware scheduler of sched21.py.

  setsid nohup ../.venv/bin/python sched22.py > ../logs/sched22_stdout.log 2>&1 &
  touch logs/SCHED22.pause | SCHED22.stop | SCHED22.reload       state logs/sched22.state.json, log logs/sched22.log

P1 build      record-first rebuild of the derived units (notes, digests) of the LME test200 + pair users: the units
              of the labelled pairs first, then the other units (3 shards) and LoCoMo         (rebuild_units.py)
P2 early      evaluations that need no new student: the v21 student on the record-first memories (comm22_t21), the
              recompute references (recompress, rebuild) and the 8B zero-shot operator (comm22_zs)
P3 labels     commutation-verified targets on the record-first memories (commutator21.py relabel, 3 shards + redo)
P4 students   tnet22 and its ablations (QLoRA on shared cards)
P5 evals      comm22 (LME S0/S1/S2 + LoCoMo + sequential recurrence), comm22_sel, comm22_overwrite, ablations
P6 bench      programmatic benchmark with per-step record correctness
P7 judges     Qwen2.5-7B on every v22 answer file; Llama-3.1-8B as the second judge of the main comparison
P8 tables     results/v22 export"""
import os

os.environ["SCHED_NAME"] = "sched22"
import sched21 as S                                   # noqa: E402

S.J.clear()
S8, S8F, S4, ST, SJ = S.S8, S.S8F, S.S4, S.ST, S.SJ
N = int(os.environ.get("SCHED_SHARDS", "3"))


def job(id_, cmd, after=(), need=(S8,)):
    S.job("v22_" + id_, cmd, after=["v22_" + x for x in after], need=need)


# The jobs are started in the order they are listed here whenever a card has room, so the list is the priority order:
# the units of the labelled pairs, their labels and the main student come first (the longest chain), the rest of the
# memories follows.
# P0: the intake of correction sessions on memories outside the test split (a few minutes on one 8B server)
job("check", "bash j22_check.sh", need=(S8,))
# P1a: the units of the training / held-out pairs; P3 labels on them.  A pair is labelled on the sentences of its unit
# as written record-first, so every pair unit is written before the labelling starts.
job("build_pairs", "bash j22_build.sh pairs", need=(S8,))
for i in range(N):
    job(f"label_{i}", f"bash j22_relabel.sh {i} {N}", after=["build_pairs"], need=(S8,))
job("label_redo", f"bash j22_relabel.sh redo {N}", after=[f"label_{i}" for i in range(N)], need=(S8F,))
job("label", f"bash j22_labelmerge.sh {N}", after=["label_redo"], need=())
# P4: main student
STUD = ["tnet22", "tnet22nd", "tnet22nv", "tnet22nr", "tnet22w"]
ROLE = {"tnet22": "main", "tnet22nd": "nd", "tnet22nv": "nv", "tnet22nr": "nr", "tnet22w": "w"}
job("train_main", "bash j22_train.sh tnet22", after=["label"], need=(ST,))
# P1b: the other units of the memories (they do not touch the pair units, so they run next to P1a wherever a card has
# room), then whatever is pending + LoCoMo + write statistics
for i in range(N):
    job(f"build_{i}", f"bash j22_build.sh lme {i} {N}", need=(S8,))
job("build_full", "bash j22_build.sh full", after=[f"build_{i}" for i in range(N)] + ["build_pairs"], need=(S8F,))
# P5: main evaluation, judged and exported as soon as it is there (the judges skip answers that carry a label and the
# export takes whatever exists, so both are run again at each later stage)
job("eval_main", "bash j22_eval.sh tnet22 main", after=["train_main", "build_full"], need=(S8, S4))
job("judge_main", "bash j22_judge.sh qwen25-7b core", after=["eval_main"], need=(SJ,))
job("tables_main", "bash j22_tables.sh", after=["judge_main"], need=())
# The benchmark was run once with both servers (11-11 01:10), so its units and its 8B rows are in the response cache;
# a run of a student needs the student alone.
job("suff_main", "bash j22_suff.sh tnet22 main cached", after=["train_main"], need=(S4,))
# P2: evaluations that need no new student.  Recompute references; the write-path control (the prose-first memories
# with attributed records and their own student, on the same two levels and the same intake); the v21 student on the
# record-first memories; the 8B zero-shot operator; the intake ablation
job("eval_recomp", "bash j22_eval.sh none recomp", after=["build_full"], need=(S8F,))
job("eval_v21", "bash j22_eval.sh tnet21 v21", after=["check", "train_main"], need=(S8, S4))
job("eval_v21sel", "bash j22_eval.sh tnet21 v21sel", after=["eval_v21"], need=(S8, S4))
job("eval_t21", "bash j22_eval.sh tnet21 t21", after=["build_full"], need=(S8, S4))
job("eval_zs", "bash j22_eval.sh none zs", after=["build_full"], need=(S8F,))
job("eval_blind", "bash j22_eval.sh tnet22 blind", after=["eval_main"], need=(S8, S4))
CORE = ["eval_main", "eval_recomp", "eval_v21", "eval_v21sel", "eval_t21", "eval_zs", "eval_blind"]
job("judge_core", "bash j22_judge.sh qwen25-7b core", after=CORE + ["judge_main"], need=(SJ,))
job("judge_llama", "bash j22_judge.sh llama31 core", after=["eval_main", "eval_t21", "eval_recomp", "eval_v21"], need=(S8,))
# direct test on the stored units (Llama-3.1-8B compares every closure unit with the recompressed reference)
job("direct_main", "bash j22_judge.sh llama31 direct comm22,comm22_sel,comm22_overwrite,comm22_zs,comm22_t21,recompress,rebuild,slots",
    after=["eval_main", "eval_t21", "eval_recomp", "eval_zs"], need=(S8,))
# the same unit test on the write-path control (attributed records), at the closure and at S(c)
job("direct_v21", "bash j22_judge.sh llama31 direct comm21np,comm21np_sel", after=["eval_v21sel", "direct_main"], need=(S8,))
# uncached single-correction latency of CoCoMem, CoCoMem-S(c) and the recompute references on the same servers
job("latency", "bash j22_latency.sh", after=["eval_main", "eval_recomp"], need=(S8F, S4))
job("suff_ref", "bash j22_suff.sh none ref", after=["suff_main"], need=(S8,))
job("tables_core", "bash j22_tables.sh", after=["judge_core", "judge_llama", "suff_main", "direct_main", "latency"], need=())
# P4b: ablation students and training seeds (one QLoRA run each), after everything the main comparison needs
# The two students of the mechanism table (no record, pointers) do not wait for the main evaluation: they are listed
# last, so they start only on a card that nothing above can use (the builds hold the evaluations back for hours).
EARLY = ("nd", "w")
for v in STUD[1:]:
    job(f"train_{ROLE[v]}", f"bash j22_train.sh {v}", after=["label", "train_main" if ROLE[v] in EARLY else "eval_main"], need=(ST,))
    job(f"eval_{ROLE[v]}", f"bash j22_eval.sh {v} {ROLE[v]}", after=[f"train_{ROLE[v]}", "eval_main"], need=(S8, S4))
    if ROLE[v] != "nv":
        job(f"suff_{ROLE[v]}", f"bash j22_suff.sh {v} {ROLE[v]} cached", after=[f"train_{ROLE[v]}"], need=(S4,))
for s in (1, 2):
    job(f"train_s{s}", f"bash j22_train.sh tnet22_s{s} {s}", after=["label", "eval_main"], need=(ST,))
    job(f"eval_s{s}", f"bash j22_eval.sh tnet22_s{s} s{s}", after=[f"train_s{s}"], need=(S8, S4))
# P7 / P8
EVALS = CORE + [f"eval_{ROLE[v]}" for v in STUD[1:]]
job("judge_qwen", "bash j22_judge.sh qwen25-7b core", after=EVALS + ["judge_core"], need=(SJ,))
job("judge_seeds", "bash j22_judge.sh qwen25-7b seeds", after=["eval_s1", "eval_s2", "judge_qwen"], need=(SJ,))
job("direct_abl", "bash j22_judge.sh llama31 direct comm22_nd,comm22_w,comm22_nv,comm22_nr",
    after=["eval_nd", "eval_w", "eval_nv", "eval_nr"], need=(S8,))
job("tables", "bash j22_tables.sh", after=["judge_qwen", "judge_llama", "judge_seeds", "suff_main", "suff_nd", "suff_nr", "suff_w", "suff_ref",
                                           "latency", "direct_main", "direct_v21", "direct_abl", "tables_core"], need=())
# the student trained without the commutation check on the programmatic benchmark (the Program columns of the ablation table)
job("suff_nv", "bash j22_suff.sh tnet22nv nv cached", after=["train_nv"], need=(S4,))
job("tables_nv", "bash j22_tables.sh", after=["suff_nv", "tables"], need=())

if __name__ == "__main__":
    os.makedirs(S.LOGS, exist_ok=True)
    S.main()
