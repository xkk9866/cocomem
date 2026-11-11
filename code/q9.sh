#!/bin/bash
# LoCoMo for the ledger-mode commutator (CPU embeddings: GPU 5 is shared), judges, final tables.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q9.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
while [ ! -e $P/logs/Q8.done ]; do sleep 60; done
log start
export CUDA_VISIBLE_DEVICES=2
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.58 >> $P/logs/q9.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 >> $P/logs/q9.log
up 18431 qwen3-8b && up 18432 tnet3 || { log SERVERS_DOWN; exit 1; }
export CM_EMB_DEVICE=cpu
for m in comm3 comm3_nofallback comm3_overwrite; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q9_$m.log 2>&1 & done
wait
log "locomo $(ls $P/results/locomo/comm3 | wc -l) $(ls $P/results/locomo/comm3_nofallback | wc -l) $(ls $P/results/locomo/comm3_overwrite | wc -l)"
bash stop_vllm.sh 18432
G="$P/results/locomo/comm3*/*.json"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q9.log
$PY judge.py --model llama31 --glob "$G" > $P/logs/q9_judge.log 2>&1
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q9.log
$PY judge.py --model qwen25-7b --glob "$G" >> $P/logs/q9_judge.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q9.log
$PY judge.py --model qwen3-8b --glob "$G" >> $P/logs/q9_judge.log 2>&1
L=comm3,comm3_nofallback,comm3_overwrite,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t8_locomo_$k.json > $P/results/t8_locomo_$k.txt 2>&1
done
$PY extras6.py > $P/results/extras6.txt 2>&1
log "ALL DONE"; touch $P/logs/Q9.done
