#!/bin/bash
# Experiment queue v3 (GPU 2 only).  Long-prompt jobs run SERIALLY in a main lane so that the prompts of
# the users in flight fit in the KV cache (prefix caching is what makes Jacobi verification cheap;
# ~175k KV tokens at 0.80 memory vs ~20k tokens per profile prompt).  Short-prompt baselines run in a
# side lane.  Idempotent: every job resumes from cached results/LLM calls.
# Log: logs/queue3.log   Done: logs/Q3.done   Launch: bash code/launch_q3.sh
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CM_EMB_DEVICE=cuda CUDA_VISIBLE_DEVICES=2 CM_CLE_DEVICE=cuda
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/queue3.log; }
up() { curl -s localhost:18431/v1/models | grep -q "\"$1\""; }
log "queue3 start"
$PY -c "import cocomem, prompts, run_corr, run_sys, locomo, judge, analyze" || { log "PREFLIGHT FAILED"; exit 1; }
up qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/queue3.log

# side lane (short prompts): Mem0/MemoryBank/RawRAG on LongMemEval, LoCoMo (persons' profiles are short)
( $PY run_sys.py --systems rawrag,membank,mem0 --splits test200 --tag main --inst 4 >> $P/logs/test_sys.log 2>&1
  CM_EMB_DEVICE=cpu $PY locomo.py run --methods spec,append,simupdate,coarse,rawrag,membank,self,recompress,rebuild,mem0 >> $P/logs/locomo_all3.log 2>&1
  log "side lane done" ) &
SIDE=$!

# main lane (serial)
[ $(ls $P/states/lineage_ncf_v2 2>/dev/null | wc -l) -ge 242 ] || $PY lineage_all.py --tag v2 >> $P/logs/lineage_all.log 2>&1
lane() { for s in "$@"; do
  $PY run_corr.py --splits test200 --tag main --strategies $s --clf_tag v2 --inst 8 --conc 64 >> $P/logs/main_$s.log 2>&1
  log "main: $s done ($(ls $P/results/main/$s | wc -l))"; done; }
lane append spec self spec_ncf &
L1=$!
lane simupdate coarse self_span ncf &
L2=$!
CM_EMB_DEVICE=cpu lane rebuild &
L3=$!
CM_EMB_DEVICE=cpu lane recompress spec_overwrite
wait $L1 $L2 $L3
$PY run_sys.py --systems oracle --splits dev,test200 --tag main --inst 4 >> $P/logs/oracle2.log 2>&1
log "main lane done"
wait $SIDE

# judges (agent-family judge, then the independent Qwen2.5 judge, which is final) and tables
$PY judge.py --model qwen3-8b --glob "$P/results/main/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_q3.log 2>&1
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.74 >> $P/logs/queue3.log
$PY judge.py --model qwen25-7b --glob "$P/results/main/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_q25.log 2>&1
log "judging finished"
M=append,simupdate,self,self_span,coarse,ncf,recompress,spec_ncf,spec_overwrite,rebuild,spec,rawrag,membank,mem0,oracle
$PY analyze.py --tag main --methods $M --ref spec --key label_qwen25-7b --json $P/results/table_main.json > $P/results/table_main.txt 2>&1
$PY analyze.py --tag main --methods $M --ref spec --key label_qwen3-8b > $P/results/table_main_q3judge.txt 2>&1
$PY analyze.py --tag main --methods $M --ref spec --key label_qwen25-7b --all > $P/results/table_main_all.txt 2>&1
$PY analyze.py --tag locomo --methods spec,append,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0 --key label_qwen25-7b --json $P/results/table_locomo.json > $P/results/table_locomo.txt 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/queue3.log
log "ALL DONE"; touch $P/logs/Q3.done
