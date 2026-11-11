#!/bin/bash
# v21 P6: judges.  $1 judge (qwen25-7b | llama31), $2 scope (core | seeds).
# qwen25-7b judges the QA answers; llama31 runs the direct test on the repaired units (comm_eval2).
source j_common.sh
JM=$1; SCOPE=${2:-core}
RC=$P/results
shopt -s nullglob
METHODS=comm21,comm21_sel,comm21_overwrite,comm21_zs,comm21_nd,comm21_w,comm21_nv,comm21_nr
if [ "$SCOPE" = seeds ]; then
  GL="$RC/main3/comm21_s1/*.json,$RC/main3/comm21_s2/*.json"
else
  GL="$RC/main3/comm21/*.json,$RC/main3/comm21_sel/*.json,$RC/main3/comm21_overwrite/*.json,$RC/main3/comm21_zs/*.json,$RC/main3/comm21_nd/*.json,$RC/main3/comm21_w/*.json,$RC/main3/comm21_nv/*.json,$RC/main3/comm21_nr/*.json,$RC/locomo/comm21/*.json,$RC/locomo/comm21_nd/*.json,$RC/locomo/comm21_nr/*.json,$RC/locomo_reg/comm21/*.json"
fi
model_path() { case $1 in qwen25-7b) echo $Q25 ;; llama31) echo $LLAMA ;; qwen3-8b) echo $Q8 ;; esac; }
B=$(budget8 $FREE); [ "$B" -gt 9000 ] && B=9000       # judge prompts are short: a 12k window is enough
J_FREE=$FREE bash start_srv.sh $G1 $P8 $(model_path $JM) $JM $(frac $B) 12288 >> $P/logs/j_servers.log || fin "JUDGE_DOWN_$JM"
if [ "$JM" = qwen25-7b ]; then
  $PY judge.py --model $JM --port $P8 --glob "$GL" >> $P/logs/q40_judge_$JM.log 2>&1 || fin "JUDGE_FAILED_$JM"
fi
if [ "$SCOPE" = core ] && [ "$JM" = llama31 ]; then
  $PY comm_eval2.py --model llama31 --port $P8 --suffix _v21 --methods $METHODS >> $P/logs/q40_comm.log 2>&1 || fin COMM_EVAL_FAILED
fi
bash stop_vllm.sh $P8
log "judge $JM $SCOPE done"
touch $P/logs/J_$J_JOB.done
