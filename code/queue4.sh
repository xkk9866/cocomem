#!/bin/bash
# Revision queue (GPU 2 only).  Stage 1 (Qwen3-8B agent): all maintenance strategies on LongMemEval-Corr with
# memory dumps (tag main3; LLM calls are cached so unchanged strategies replay quickly), new strategies
# ccf / slots / ccf_overwrite, LoCoMo with ccf+slots.  Stage 2: Llama-3.1-8B (third family) judges QA and
# runs the direct commutativity evaluation.  Stage 3: Qwen2.5-7B judge.  Then tables.  Log: logs/queue4.log
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/queue4.log; }
up() { curl -s localhost:18431/v1/models | grep -q "\"$1\""; }
log start
$PY -c "import cocomem, run_corr, locomo, comm_eval, judge, analyze" || { log PREFLIGHT_FAILED; exit 1; }
up qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/queue4.log
for s in rawrag membank mem0 oracle; do mkdir -p $P/results/main3; [ -e $P/results/main3/$s ] || cp -r $P/results/main/$s $P/results/main3/$s; done
lane() { for s in "$@"; do
  $PY run_corr.py --splits test200 --tag main3 --strategies $s --clf $P/models/ccf --clf_tag v2 --inst 10 --conc 64 >> $P/logs/m3_$s.log 2>&1
  log "main3 $s $(ls $P/results/main3/$s | wc -l)"; done; }
lane recompress ccf slots append &
lane coarse self_span simupdate ccf_overwrite &
lane ncf spec rebuild self &
$PY locomo.py run --methods ccf,slots --clf $P/models/ccf >> $P/logs/locomo_v3.log 2>&1 &
wait
log "stage1 done"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/queue4.log
$PY comm_eval.py --tag main3 --model llama31 > $P/logs/comm_eval.log 2>&1
$PY judge.py --model llama31 --glob "$P/results/main3/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_llama.log 2>&1
log "stage2 done"
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/queue4.log
$PY judge.py --model qwen25-7b --glob "$P/results/main3/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_q25_v3.log 2>&1
log "stage3 done"
M=append,slots,simupdate,self,self_span,coarse,ncf,ccf,recompress,spec,rebuild,ccf_overwrite,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31; do
  $PY analyze.py --tag main3 --methods $M --ref ccf --key label_$k --json $P/results/t3_main_$k.json > $P/results/t3_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods ccf,slots,ncf,spec,append,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem --key label_$k --json $P/results/t3_locomo_$k.json > $P/results/t3_locomo_$k.txt 2>&1
done
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/queue4.log
$PY extras.py --tag main3 > $P/results/extras.txt 2>&1
log "ALL DONE"; touch $P/logs/Q4.done
