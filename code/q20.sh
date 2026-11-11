#!/bin/bash
# finish the recurrence test: wait for the baseline runs, judge them (Qwen2.5-7B, GPU 2), analyse.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q20.log; }
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
log start
while pgrep -f 'locomo.py run --regress' > /dev/null; do sleep 30; done
log "regress runs finished: $(for d in $P/results/locomo_reg/*/; do echo -n "$(basename $d)=$(ls $d | grep -c 'json$') "; done)"
for p in 18431 18432; do bash stop_vllm.sh $p; done; sleep 5
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.85 >> $P/logs/q20.log
$PY judge.py --model qwen25-7b --glob "$P/results/locomo_reg/*/*.json" >> $P/logs/q20_judge.log 2>&1
bash stop_vllm.sh 18431
$PY reg19.py comm12,comm12T,ncf,coarse,rebuild,append > $P/results/reg19.txt 2>&1
[ -s $P/results/reg19.json ] && log "reg19 done" || log "REG19_FAILED"
cd $P/results && tar czf $P/r20.tgz reg19.json reg19.txt && log packed
touch $P/logs/Q20.done
