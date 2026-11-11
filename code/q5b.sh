#!/bin/bash
# v4 stage B (GPU 2 only).  Distil the commutator, serve it int4 beside the compressor, evaluate it,
# run the v4 strategies on LongMemEval-Corr + LoCoMo-Corr, direct commutativity test, three judges, tables.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cuda
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q5.log; }
while [ ! -e $P/logs/Q5A.done ]; do sleep 60; done
bash stop_vllm.sh 18432
log "B start"
$PY -c "import cocomem, run_corr, locomo, commutator, comm_eval2, jacobi2, analyze" || { log PREFLIGHT_FAILED; exit 1; }
if [ ! -e $P/models/tnet/merged/config.json ]; then
  $PY commutator.py data >> $P/logs/q5_tnet.log 2>&1
  bash stop_vllm.sh 18431
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train >> $P/logs/q5_tnet.log 2>&1
  log "tnet trained"
fi
[ -e $P/models/tnet/merged/config.json ] || { log "TNET_MISSING abort"; exit 1; }
[ -e $P/results/tnet_score.json ] || $PY commutator.py score >> $P/logs/q5_tnet.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.62 >> $P/logs/q5.log
bash start_tnet.sh >> $P/logs/q5.log
log "servers up"
$PY commutator.py eval --served tnet --port 18432 >> $P/logs/q5_tnet.log 2>&1
$PY commutator.py eval --served qwen3-8b --port 18431 >> $P/logs/q5_tnet.log 2>&1
log "tnet eval done"
$PY jacobi2.py > $P/logs/q5_jacobi2.log 2>&1 &
lane() { for s in "$@"; do
  $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 96 >> $P/logs/m3_$s.log 2>&1
  log "main3 $s $(ls $P/results/main3/$s | wc -l)"; done; }
lane comm comm_overwrite &
lane comm_zs &
$PY run_sys.py --systems oracle --splits test200 --tag main3 --inst 8 >> $P/logs/m3_oracle.log 2>&1 &
$PY locomo.py run --methods comm,comm_zs,comm_overwrite >> $P/logs/locomo_v4.log 2>&1 &
wait
log "stage1 done"
bash stop_vllm.sh 18432
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q5.log
$PY comm_eval2.py --model llama31 > $P/logs/q5_comm2.log 2>&1
$PY judge.py --model llama31 --glob "$P/results/main3/comm*/*.json,$P/results/main3/oracle/*.json,$P/results/locomo/comm*/*.json" > $P/logs/q5_judge_llama.log 2>&1
log "llama done"
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q5.log
$PY judge.py --model qwen25-7b --glob "$P/results/main3/comm*/*.json,$P/results/main3/oracle/*.json,$P/results/locomo/comm*/*.json" > $P/logs/q5_judge_q25.log 2>&1
log "qwen25 done"
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/q5.log
$PY judge.py --model qwen3-8b --glob "$P/results/main3/comm*/*.json,$P/results/main3/oracle/*.json,$P/results/locomo/comm*/*.json" > $P/logs/q5_judge_q3.log 2>&1
M=append,slots,simupdate,self,self_span,coarse,ncf,ccf,recompress,spec,rebuild,comm_zs,comm,comm_overwrite,rawrag,membank,mem0,oracle
L=comm,comm_zs,comm_overwrite,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm --key label_$k --json $P/results/t5_main_$k.json > $P/results/t5_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t5_locomo_$k.json > $P/results/t5_locomo_$k.txt 2>&1
done
log "ALL DONE"; touch $P/logs/Q5B.done
