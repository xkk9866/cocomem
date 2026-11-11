#!/bin/bash
# v22 P7: judges.  $1 judge (qwen25-7b | llama31), $2 scope (core | seeds | direct), $3 methods (direct only).
#   qwen25-7b core     every v22 answer file (LME main3, LoCoMo, LoCoMo recurrence)
#   llama31 core       second judge of the main comparison: CoCoMem (v22, v21, recompute references) and the six
#                      published systems, LME and LoCoMo, so that judge agreement can be reported
#   llama31 direct M   direct test on the stored units (comm_eval2.py): every unit of the closure of an evidence
#                      correction is compared with the recompressed reference; Sound / Stale per unit
#   qwen25-7b seeds    the seed students
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
JM=$1; SCOPE=${2:-core}; METHODS=${3:-}
R0=$P/results; RC=$CM_RUN_ROOT/results
shopt -s nullglob
mkdir -p $RC/main3 $RC/locomo
for b in rawrag readagent amem ga membank mem0 oracle comm21 comm21_sel comm21_overwrite comm21_zs comm21_nd comm21_w comm21_nv comm21_nr; do
  [ -e $RC/main3/$b ] || [ ! -d $R0/main3/$b ] || ln -s $R0/main3/$b $RC/main3/$b
  [ -e $RC/locomo/$b ] || [ ! -d $R0/locomo/$b ] || ln -s $R0/locomo/$b $RC/locomo/$b
done
if [ "$SCOPE" = seeds ]; then
  GL="$RC/main3/comm22_s1/*.json,$RC/main3/comm22_s2/*.json"
elif [ "$SCOPE" = direct ]; then
  GL=""
elif [ "$JM" = llama31 ]; then
  GL="$RC/main3/comm22/*.json,$RC/main3/comm22_t21/*.json,$RC/main3/comm21np/*.json,$RC/main3/recompress/*.json,$RC/main3/rebuild/*.json,$RC/locomo/comm22/*.json,$RC/locomo/comm21np/*.json"
  for b in rawrag readagent amem ga membank mem0 comm21; do GL="$GL,$R0/main3/$b/*.json,$R0/locomo/$b/*.json"; done
else
  GL="$RC/main3/*/*.json,$RC/locomo/*/*.json,$RC/locomo_reg/*/*.json"
fi
model_path() { case $1 in qwen25-7b) echo $Q25 ;; llama31) echo $LLAMA ;; qwen3-8b) echo $Q8 ;; esac; }
B=$(budget8 $FREE); [ "$B" -gt 9000 ] && B=9000       # judge prompts are short: a 12k window is enough
J_FREE=$FREE bash start_srv.sh $G1 $P8 $(model_path $JM) $JM $(frac $B) 12288 >> $P/logs/j_servers.log || fin "JUDGE_DOWN_$JM"
if [ "$SCOPE" = direct ]; then
  [ -n "$METHODS" ] || fin DIRECT_NO_METHODS
  $PY comm_eval2.py --model $JM --port $P8 --suffix _v22_$J_JOB --methods $METHODS >> $P/logs/v22_direct.log 2>&1 || fin COMM_EVAL_FAILED
else
  SEC=""; [ "$JM" = llama31 ] && SEC="--secondary"
  $PY judge.py --model $JM --port $P8 --glob "$GL" $SEC >> $P/logs/v22_judge_$JM.log 2>&1 || fin "JUDGE_FAILED_$JM"
fi
bash stop_vllm.sh $P8
log "judge $JM $SCOPE done"
touch $P/logs/J_$J_JOB.done
