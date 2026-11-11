#!/bin/bash
# Stage B (GPU 2 only): train NCF -> lineage quality -> write-time NCF index -> all correction experiments.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/stageB.log; }
# wait for the counterfactual self-supervision
sleep 60
while pgrep -f "cfgen.py --splits" > /dev/null; do sleep 20; done
log "cfgen finished: $(wc -l < $P/data/cf_dev.jsonl) dev rows, $(wc -l < $P/data/cf_test.jsonl) test rows"
if [ ! -f $P/models/ncf/ncf.pt ]; then
  kill $(cat $P/logs/vllm_cur.pid) 2>/dev/null
  L=$(ss -ltnp 2>/dev/null | grep ":18431 " | grep -o 'pid=[0-9]*' | cut -d= -f2); [ -n "$L" ] && kill $L; sleep 20
  log "train NCF"
  CUDA_VISIBLE_DEVICES=2 $PY ncf.py train --data $P/data/cf_dev.jsonl --out $P/models/ncf > $P/logs/ncf_train.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $PY ncf.py eval --data $P/data/cf_test.jsonl --out $P/models/ncf --report $P/results/lineage_quality.json > $P/logs/ncf_eval.log 2>&1
  log "NCF trained/evaluated"
fi
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/stageB.log
CUDA_VISIBLE_DEVICES=2 CM_CLE_DEVICE=cuda $PY lineage_all.py > $P/logs/lineage_all.log 2>&1
log "NCF index built"
export CM_EMB_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=2 CM_CLE_DEVICE=cuda
for s in ncf coarse self simupdate append rebuild ncf_unit ncf_nofacts ncf_overwrite self_span; do
  $PY run_corr.py --splits test200 --tag main --strategies $s --inst 6 --conc 64 > $P/logs/main_$s.log 2>&1 &
done
$PY run_sys.py --systems oracle --splits dev,test200 --tag main --inst 8 >> $P/logs/oracle2.log 2>&1 &
$PY run_sys.py --systems mem0 --splits test200 --tag main --inst 10 >> $P/logs/test_sys.log 2>&1 &
$PY locomo.py run --methods ncf,coarse,self,simupdate,append,rebuild > $P/logs/locomo_ours.log 2>&1 &
$PY locomo.py run --methods rawrag,membank,mem0,amem > $P/logs/locomo_base.log 2>&1 &
wait
log "stage B runs finished"
touch $P/logs/B.done
