#!/bin/bash
# Completes v6 tables: LoCoMo for the sufficiency-aware commutator, no-cache latency for every strategy, judges.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q8.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
log start
export CUDA_VISIBLE_DEVICES=2
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.58 >> $P/logs/q8.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 >> $P/logs/q8.log
up 18431 qwen3-8b && up 18432 tnet3 || { log SERVERS_DOWN; exit 1; }
export CUDA_VISIBLE_DEVICES=5 CM_EMB_DEVICE=cuda
rm -rf $P/results/locomo/comm3 $P/results/locomo/comm3_nofallback $P/results/locomo/comm3_overwrite
$PY locomo.py run --methods comm3 >> $P/logs/locomo_v6.log 2>&1 &
$PY locomo.py run --methods comm3_nofallback >> $P/logs/locomo_v6b.log 2>&1 &
$PY locomo.py run --methods comm3_overwrite >> $P/logs/locomo_v6c.log 2>&1 &
wait
log "locomo $(ls $P/results/locomo/comm3 | wc -l) $(ls $P/results/locomo/comm3_nofallback | wc -l) $(ls $P/results/locomo/comm3_overwrite | wc -l)"
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm3,ncf,coarse,recompress,spec,rebuild,simupdate,slots --out latency_a.json > $P/logs/q8_lat_a.log 2>&1
export CUDA_VISIBLE_DEVICES=2
bash stop_vllm.sh 18432
bash start_tnet.sh $P/models/tnet2/merged 0.20 tnet2 >> $P/logs/q8.log
CUDA_VISIBLE_DEVICES=5 CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm2 --out latency_b.json > $P/logs/q8_lat_b.log 2>&1
log "latency $(ls $P/results/latency_*.json | wc -l)"
bash stop_vllm.sh 18432
G="$P/results/locomo/comm3*/*.json"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q8.log
$PY judge.py --model llama31 --glob "$G" > $P/logs/q8_judge_llama.log 2>&1
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q8.log
$PY judge.py --model qwen25-7b --glob "$G" > $P/logs/q8_judge_q25.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q8.log
$PY judge.py --model qwen3-8b --glob "$G" > $P/logs/q8_judge_q3.log 2>&1
L=comm3,comm3_nofallback,comm3_overwrite,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t8_locomo_$k.json > $P/results/t8_locomo_$k.txt 2>&1
done
CM_EMB_DEVICE=cpu $PY extras6.py > $P/results/extras6.txt 2>&1
log "ALL DONE"; touch $P/logs/Q8.done
