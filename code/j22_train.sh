#!/bin/bash
# v22 P4: one student (Qwen3-4B, LoRA r16, 1 epoch, effective batch 16) on this job's GPU.  $1 variant, $2 seed.
# Same recipe as j_train.sh (QLoRA when the shared card has little free memory; adapter merged into the bf16 base).
source j_common.sh
export CM_RUN_ROOT=$P/v22
trap - EXIT
V=$1; SEED=${2:-0}
ML=2048; case $V in *w) ML=3072 ;; esac                                            # pointer witnesses are long
if [ "$FREE" -ge 15000 ]; then Q=""; TB=4096; [ "$FREE" -lt 22000 ] && TB=3072
else Q="--qlora"; TB=2048; [ "$FREE" -lt 9000 ] && TB=1536; [ "$FREE" -lt 8000 ] && TB=1024; fi
[ -e $P/models/$V/merged/config.json ] || rm -f $P/models/$V/train_meta.json      # a crash between merge and save
A=$P/models/$V/adapter
if [ ! -e $P/models/$V/train_meta.json ]; then
  [ -s $P/models/$V/train.jsonl ] || fin "NO_DATA_$V"
  if [ -s $A/train_stats.json ] && [ -s $A/adapter_model.safetensors ] && [ $A/adapter_model.safetensors -nt $P/models/$V/train.jsonl ]; then
    # the run finished and its adapter is on disk; only the merge into the bf16 base is missing (CPU)
    rm -rf $P/models/$V/merged
    CUDA_VISIBLE_DEVICES="" $PY commutator21.py train --out $P/models/$V --max_len $ML --seed $SEED --merge-only >> $P/logs/v22_train_$V.log 2>&1 || fin "MERGE_FAILED_$V"
  else
    rm -rf $P/models/$V/merged $A
    # The response cache of a student is keyed by its name: the answers of the weights that are being replaced must
    # not be served for the new ones (11-11: the programmatic benchmark of the retrained student came back from the
    # cache of the first one, number for number).  The cache is set aside with the weights it belongs to.
    if [ -e $P/cache/$V.sqlite ]; then
      mkdir -p $P/cache/old; TS=$(date +%m%d_%H%M%S)
      for x in "" -wal -shm; do
        if [ -e $P/cache/$V.sqlite$x ]; then mv $P/cache/$V.sqlite$x $P/cache/old/$V.$TS.sqlite$x; fi
      done
      log "response cache of $V set aside (cache/old/$V.$TS.sqlite)"
    fi
    CUDA_VISIBLE_DEVICES=$G1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY commutator21.py train --out $P/models/$V --max_len $ML --tok_budget $TB --seed $SEED $Q >> $P/logs/v22_train_$V.log 2>&1 || fin "TRAIN_FAILED_$V (tok_budget=$TB $Q free=$FREE)"
  fi
fi
[ -e $P/models/$V/merged/config.json ] && [ -e $P/models/$V/train_meta.json ] || fin "MISSING_MODEL_$V"
mkdir -p $CM_RUN_ROOT/results
cp $P/models/$V/train_meta.json $CM_RUN_ROOT/results/train_meta_$V.json
log "trained $V seed=$SEED tok_budget=$TB $Q free=$FREE: $(tail -2 $P/logs/v22_train_$V.log | tr '\n' ' ' | cut -c1-200)"
touch $P/logs/J_$J_JOB.done
