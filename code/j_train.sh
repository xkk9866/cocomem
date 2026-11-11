#!/bin/bash
# v21 P3: one student (Qwen3-4B, LoRA r16, 1 epoch, effective batch 16) on this job's GPU.  $1 variant, $2 seed.
# On a shared card with a few GB free the frozen base is held in 4-bit (QLoRA) and the adapter is merged into the
# bf16 base afterwards; the LoRA recipe itself (rank, alpha, lr, batch, epochs, data) is identical.
source j_common.sh
trap - EXIT
V=$1; SEED=${2:-0}
ML=2048; [ "$V" = tnet21w ] && ML=3072                                             # pointer witnesses are long
if [ "$FREE" -ge 15000 ]; then Q=""; TB=4096; [ "$FREE" -lt 22000 ] && TB=3072
else Q="--qlora"; TB=2048; [ "$FREE" -lt 9000 ] && TB=1536; [ "$FREE" -lt 8000 ] && TB=1024; fi
[ -e $P/models/$V/merged/config.json ] || rm -f $P/models/$V/train_meta.json      # a crash between merge and save
if [ ! -e $P/models/$V/train_meta.json ]; then
  [ -s $P/models/$V/train.jsonl ] || fin "NO_DATA_$V"
  rm -rf $P/models/$V/merged $P/models/$V/adapter
  CUDA_VISIBLE_DEVICES=$G1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY commutator21.py train --out $P/models/$V --max_len $ML --tok_budget $TB --seed $SEED $Q >> $P/logs/q40_train_$V.log 2>&1 || fin "TRAIN_FAILED_$V (tok_budget=$TB $Q free=$FREE)"
fi
[ -e $P/models/$V/merged/config.json ] && [ -e $P/models/$V/train_meta.json ] || fin "MISSING_MODEL_$V"
cp $P/models/$V/train_meta.json $P/results/train_meta_$V.json
log "trained $V seed=$SEED tok_budget=$TB $Q free=$FREE: $(tail -2 $P/logs/q40_train_$V.log | tr '\n' ' ' | cut -c1-200)"
touch $P/logs/J_$J_JOB.done
