#!/bin/bash
# v5 (GPU 2 servers; GPU 5 embedding clients): information-level commutator distillation.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q6.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
log "start"
export CUDA_VISIBLE_DEVICES=2
up 18431 qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q6.log
if [ ! -e $P/models/tnet2/test.jsonl ]; then
  $PY commutator.py relabel --src $P/models/tnet --out $P/models/tnet2 >> $P/logs/q6_tnet.log 2>&1
fi
[ -e $P/models/tnet2/test.jsonl ] || { log "RELABEL_FAILED"; exit 1; }
log "relabel $(grep -E '^(train|test) ' $P/logs/q6_tnet.log | tail -2 | tr '\n' ' ')"
if [ ! -e $P/models/tnet2/merged/config.json ]; then
  bash stop_vllm.sh 18431
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train --out $P/models/tnet2 >> $P/logs/q6_tnet.log 2>&1
fi
[ -e $P/models/tnet2/merged/config.json ] || { log "TRAIN_FAILED"; exit 1; }
log "trained"
up 18431 qwen3-8b && bash stop_vllm.sh 18431
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.58 >> $P/logs/q6.log
bash start_tnet.sh $P/models/tnet2/merged 0.20 tnet2 >> $P/logs/q6.log
up 18431 qwen3-8b && up 18432 tnet2 || { log "SERVERS_DOWN"; exit 1; }
log "servers up"
$PY commutator.py eval --out $P/models/tnet2 --served tnet2 --port 18432 --suffix _v5 >> $P/logs/q6_tnet.log 2>&1
$PY commutator.py eval --out $P/models/tnet2 --served qwen3-8b --port 18431 --suffix _v5 >> $P/logs/q6_tnet.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q6_tnet.log | tail -2 | cut -c1-120 | tr '\n' ' ')"
export CUDA_VISIBLE_DEVICES=5 CM_EMB_DEVICE=cuda
$PY run_corr.py --splits test200 --tag main3 --strategies comm2 --inst 12 --conc 128 >> $P/logs/m3_comm2.log 2>&1 &
$PY run_corr.py --splits test200 --tag main3 --strategies comm2_overwrite --inst 12 --conc 128 >> $P/logs/m3_comm2o.log 2>&1 &
$PY locomo.py run --methods comm2,comm2_overwrite >> $P/logs/locomo_v5.log 2>&1 &
wait
for s in comm2 comm2_overwrite; do log "main3 $s $(ls $P/results/main3/$s | wc -l) locomo $(ls $P/results/locomo/$s 2>/dev/null | wc -l)"; done
export CUDA_VISIBLE_DEVICES=2
bash stop_vllm.sh 18432; bash stop_vllm.sh 18431
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q6.log
$PY comm_eval2.py --model llama31 --suffix _v5 \
  --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm_overwrite,comm2,comm2_overwrite,recompress,rebuild > $P/logs/q6_comm2.log 2>&1
G="$P/results/main3/comm2*/*.json,$P/results/locomo/comm2*/*.json"
$PY judge.py --model llama31 --glob "$G" > $P/logs/q6_judge_llama.log 2>&1
log "llama done"
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q6.log
$PY judge.py --model qwen25-7b --glob "$G" > $P/logs/q6_judge_q25.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q6.log
$PY judge.py --model qwen3-8b --glob "$G" > $P/logs/q6_judge_q3.log 2>&1
log "judges done"
M=append,slots,simupdate,self,self_span,coarse,ncf,ccf,recompress,spec,rebuild,comm_zs,comm,comm_overwrite,comm2,comm2_overwrite,rawrag,membank,mem0,oracle
L=comm2,comm2_overwrite,comm,comm_zs,comm_overwrite,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm2 --key label_$k --json $P/results/t6_main_$k.json > $P/results/t6_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t6_locomo_$k.json > $P/results/t6_locomo_$k.txt 2>&1
done
log "ALL DONE"; touch $P/logs/Q6.done
