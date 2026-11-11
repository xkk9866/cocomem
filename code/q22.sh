#!/bin/bash
# q22: post-processing of the OFFICIAL baselines.  Start it after all official_run.py processes have finished.
#   1. reader (Qwen3-8B, the shared server on 18433) answers from the stored contexts      (official_read.py)
#   2. three judges grade the answers                                                       (judge.py)
#   3. tables and statistics (all methods, common question set)                             (analyze.py, stat22.py)
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q22.log; }
export CUDA_VISIBLE_DEVICES= CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
log start
SYS=${SYS:-"mem0 amem langmem"}
for s in $SYS; do
  TAGL=main; TAGM=main; [ $s = amem ] && TAGM=main_s
  [ -d $P/bl/out/$TAGM/$s ] && CM_P8=18433 $PY official_read.py --system $s --task lme --tag $TAGM >> $P/logs/q22_read.log 2>&1
  [ -d $P/bl/out/$TAGL/$s ] && CM_P8=18433 $PY official_read.py --system $s --task locomo --tag $TAGL >> $P/logs/q22_read.log 2>&1
done
log "read done: $(ls $P/results/main3 | grep off_ | tr '\n' ' ')"
touch $P/logs/KEEP_BL.stop
bash stop_vllm.sh 18433; ps aux | grep '[l]lm_proxy' | awk '{print $2}' | xargs -r kill
unset CUDA_VISIBLE_DEVICES; export CUDA_VISIBLE_DEVICES=2
G="$P/results/main3/off_*/*.json,$P/results/locomo/off_*/*.json,$P/results/locomo_reg/off_*/*.json"
for jm in "qwen25-7b /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ" "qwen3-8b /home/yaokang/models/Qwen3-8B-AWQ" "llama31 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ"; do
  set -- $jm
  bash start_gpu2.sh $2 $1 0.85 >> $P/logs/q22.log
  curl -s localhost:18431/v1/models | grep -q "\"$1\"" || { log "JUDGE_DOWN_$1"; continue; }
  CM_EMB_DEVICE=cpu $PY judge.py --model $1 --glob "$G" >> $P/logs/q22_judge.log 2>&1
done
bash stop_vllm.sh 18431
log judged
#touch $P/logs/Q22.done
# ---- tables and statistics (the sets differ because the official baselines were stopped at different points)
unset CUDA_VISIBLE_DEVICES; export CUDA_VISIBLE_DEVICES= CM_EMB_DEVICE=cpu
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods rawrag,membank,off_mem0,comm10,comm11,oracle --ref comm10 --key label_$k --json $P/results/t22M_main_$k.json > $P/results/t22M_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods rawrag,membank,off_amem,comm10,comm11,oracle --ref comm10 --key label_$k --json $P/results/t22A_main_$k.json > $P/results/t22A_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods rawrag,membank,off_langmem,comm10,comm11,oracle --ref comm10 --key label_$k --json $P/results/t22L_main_$k.json > $P/results/t22L_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods rawrag,membank,off_mem0,off_amem,off_langmem,comm10,comm11,comm_zs,ncf,coarse,rebuild,append --ref comm10 --key label_$k --json $P/results/t22_locomo_$k.json > $P/results/t22_locomo_$k.txt 2>&1
  for b in M:off_mem0 A:off_amem L:off_langmem; do
    tag=${b%%:*}; m=${b##*:}
    $PY stat19.py comm10,comm11 rawrag,membank,$m label_$k > /dev/null 2>&1; cp $P/results/stat19$( [ $k = qwen25-7b ] && echo "" || echo "_$k" ).json $P/results/stat22${tag}_$k.json
  done
done
$PY reg19.py comm12,comm10,off_mem0,off_amem,off_langmem,rawrag,membank > $P/results/reg22.txt 2>&1
cd $P/results && tar czf $P/r22.tgz t22*.json stat22*.json reg19.json coverage_*.json cov_*.json 2>/dev/null
log "ALL DONE"
touch $P/logs/Q22.done
