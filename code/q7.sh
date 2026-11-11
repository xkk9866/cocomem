#!/bin/bash
# v6: sufficiency-aware commutator (tnet3) + ledger fallback.  GPU 2: servers; GPU 5: embedding clients.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q7.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
log "start"
export CUDA_VISIBLE_DEVICES=2
bash stop_vllm.sh 18432
up 18431 qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q7.log
[ -e $P/models/tnet3/test.jsonl ] || $PY commutator.py relabel3 --src $P/models/tnet2 --out $P/models/tnet3 >> $P/logs/q7_tnet.log 2>&1
[ -e $P/models/tnet3/test.jsonl ] || { log "RELABEL3_FAILED"; exit 1; }
log "relabel3 $(grep -E '^(train|test) ' $P/logs/q7_tnet.log | tail -2 | tr '\n' ' ')"
if [ ! -e $P/models/tnet3/merged/config.json ]; then
  bash stop_vllm.sh 18431
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train --prompt 3 --out $P/models/tnet3 >> $P/logs/q7_tnet.log 2>&1
fi
[ -e $P/models/tnet3/merged/config.json ] || { log "TRAIN_FAILED"; exit 1; }
log "trained"
up 18431 qwen3-8b && bash stop_vllm.sh 18431
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.58 >> $P/logs/q7.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 >> $P/logs/q7.log
up 18431 qwen3-8b && up 18432 tnet3 || { log "SERVERS_DOWN"; exit 1; }
log "servers up"
$PY commutator.py eval --prompt 3 --out $P/models/tnet3 --served tnet3 --port 18432 --suffix _v6 >> $P/logs/q7_tnet.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q7_tnet.log | tail -1 | cut -c1-200)"
export CUDA_VISIBLE_DEVICES=5 CM_EMB_DEVICE=cuda
for s in comm3 comm3_nofallback comm3_overwrite; do
  $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 12 --conc 128 >> $P/logs/m3_$s.log 2>&1 &
done
$PY locomo.py run --methods comm3,comm3_nofallback,comm3_overwrite >> $P/logs/locomo_v6.log 2>&1 &
wait
for s in comm3 comm3_nofallback comm3_overwrite; do log "main3 $s $(ls $P/results/main3/$s | wc -l) locomo $(ls $P/results/locomo/$s 2>/dev/null | wc -l)"; done
CM_NOCACHE=1 $PY bench_latency.py --n 30 > $P/logs/q7_latency.log 2>&1
log "latency done"
export CUDA_VISIBLE_DEVICES=2
bash stop_vllm.sh 18432; bash stop_vllm.sh 18431
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q7.log
$PY comm_eval2.py --model llama31 --suffix _v6 \
  --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm2,comm3,comm3_nofallback,comm3_overwrite,recompress,rebuild > $P/logs/q7_comm2.log 2>&1
G="$P/results/main3/comm3*/*.json,$P/results/locomo/comm3*/*.json"
$PY judge.py --model llama31 --glob "$G" > $P/logs/q7_judge_llama.log 2>&1
log "llama done"
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q7.log
$PY judge.py --model qwen25-7b --glob "$G" > $P/logs/q7_judge_q25.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q7.log
$PY judge.py --model qwen3-8b --glob "$G" > $P/logs/q7_judge_q3.log 2>&1
log "judges done"
M=append,slots,simupdate,self,self_span,coarse,ncf,ccf,recompress,spec,rebuild,comm_zs,comm,comm2,comm3_nofallback,comm3_overwrite,comm3,rawrag,membank,mem0,oracle
L=comm3,comm3_nofallback,comm3_overwrite,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm3 --key label_$k --json $P/results/t7_main_$k.json > $P/results/t7_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t7_locomo_$k.json > $P/results/t7_locomo_$k.txt 2>&1
done
CM_EMB_DEVICE=cpu $PY extras6.py > $P/results/extras6.txt 2>&1
log "ALL DONE"; touch $P/logs/Q7.done
