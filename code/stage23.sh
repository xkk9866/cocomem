#!/bin/bash
# Stage 2-3 of the revision queue: third-family judge (Llama-3.1-8B int4) + Qwen2.5 judge + tables.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/queue4.log; }
M=/data/yaokang/models/Llama-3.1-8B-Instruct-AWQ
while [ ! -f $M/model-00001-of-00002.safetensors ] || pgrep -f "huggingface-cli download" > /dev/null; do sleep 30; done
log "llama weights complete"
bash start_gpu2.sh $M llama31 0.80 >> $P/logs/queue4.log
curl -s localhost:18431/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"llama31","messages":[{"role":"user","content":"Is 2+2=4? Answer yes or no."}],"max_tokens":3}' >> $P/logs/queue4.log
$PY comm_eval.py --tag main3 --model llama31 > $P/logs/comm_eval.log 2>&1
$PY judge.py --model llama31 --glob "$P/results/main3/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_llama.log 2>&1
log "stage2 done"
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/queue4.log
$PY judge.py --model qwen25-7b --glob "$P/results/main3/*/*.json,$P/results/locomo/*/*.json" > $P/logs/judge_q25_v3.log 2>&1
log "stage3 done"
MM=append,slots,simupdate,self,self_span,coarse,ncf,ccf,recompress,spec,rebuild,ccf_overwrite,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $MM --ref ccf --key label_$k --json $P/results/t3_main_$k.json > $P/results/t3_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods ccf,slots,ncf,spec,append,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem --key label_$k --json $P/results/t3_locomo_$k.json > $P/results/t3_locomo_$k.txt 2>&1
done
$PY extras.py --tag main3 > $P/results/extras.txt 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/queue4.log
log "ALL DONE"; touch $P/logs/Q4.done
