#!/bin/bash
# Full experiment queue v2 (GPU 2 only).  Idempotent: every job resumes from cached LLM calls/results.
# Log: logs/queue2.log   Done marker: logs/Q2.done   Re-run: bash code/launch_q2.sh
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CM_EMB_DEVICE=cpu CUDA_VISIBLE_DEVICES=2 CM_CLE_DEVICE=cuda
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/queue2.log; }
busy() { pgrep -f "run_corr.py|run_sys.py|locomo.py run|lineage_all.py|judge.py" > /dev/null; }
up() { curl -s localhost:18431/v1/models | grep -q "\"$1\""; }
log "queue2 start"
up qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/queue2.log

# ---- phase 1: all systems (parallel on one server)
( $PY lineage_all.py --tag v2 > $P/logs/lineage_all.log 2>&1
  for s in spec_ncf ncf; do $PY run_corr.py --splits test200 --tag main --strategies $s --clf_tag v2 --inst 6 --conc 64 > $P/logs/main_$s.log 2>&1; done ) &
for s in spec recompress spec_overwrite coarse self self_span rebuild simupdate append; do
  pgrep -f "strategies $s " > /dev/null || $PY run_corr.py --splits test200 --tag main --strategies $s --inst 6 --conc 64 >> $P/logs/main_$s.log 2>&1 &
done
pgrep -f "systems oracle" > /dev/null || $PY run_sys.py --systems oracle --splits dev,test200 --tag main --inst 8 >> $P/logs/oracle2.log 2>&1 &
pgrep -f "systems mem0" > /dev/null || $PY run_sys.py --systems rawrag,membank,mem0 --splits test200 --tag main --inst 10 >> $P/logs/test_sys.log 2>&1 &
pgrep -f "locomo.py run --methods rawrag" > /dev/null || $PY locomo.py run --methods rawrag,membank,mem0 >> $P/logs/locomo_base.log 2>&1 &
$PY locomo.py run --methods spec,append,simupdate,coarse,self,recompress,rebuild > $P/logs/locomo_ours.log 2>&1 &
sleep 120
while busy; do sleep 60; done
log "phase 1 finished: $(for d in $P/results/main/*; do echo -n "$(basename $d)=$(ls $d | wc -l) "; done)"

# ---- phase 2: judges (agent-family judge first, then the independent Qwen2.5 judge, which is final)
up qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/queue2.log
$PY judge.py --model qwen3-8b --glob "$P/results/main/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_q3.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.85 >> $P/logs/queue2.log
$PY judge.py --model qwen25-7b --glob "$P/results/main/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_q25.log 2>&1
log "judging finished"

# ---- phase 3: tables
M=append,simupdate,self,self_span,coarse,ncf,recompress,spec_ncf,spec_overwrite,rebuild,spec,rawrag,membank,mem0,oracle
$PY analyze.py --tag main --methods $M --ref spec --key label_qwen25-7b --json $P/results/table_main.json > $P/results/table_main.txt 2>&1
$PY analyze.py --tag main --methods $M --ref spec --key label_qwen3-8b > $P/results/table_main_q3judge.txt 2>&1
$PY analyze.py --tag main --methods $M --ref spec --key label_qwen25-7b --all > $P/results/table_main_all.txt 2>&1
$PY analyze.py --tag locomo --methods spec,append,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem --key label_qwen25-7b --json $P/results/table_locomo.json > $P/results/table_locomo.txt 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/queue2.log
log "ALL DONE"; touch $P/logs/Q2.done
