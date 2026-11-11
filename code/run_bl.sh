#!/bin/bash
# run_bl.sh SYSTEM TASK PROCS [TAG] [SPLITS] [LIMIT]: one official baseline on one task (env per system)
P=/data/yaokang/corrmem; cd $P/code
S=$1; T=$2; N=$3; TAG=${4:-main}; SP=${5:-test200}; LIM=${6:-0}
case $S in mem0|amem) V=venv_m0;; mem0v2) V=venv_m2;; langmem) V=venv_lm;; esac
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false BL_PROXY=${BL_PROXY:-18440} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$P/bl/$V/bin/python official_run.py --system $S --task $T --splits $SP --tag $TAG --procs $N --limit $LIM 2>&1 | grep -v -E "Loading weights|PostHog" >> $P/logs/bl_${S}_${T}_${TAG}.log
echo done >> $P/logs/bl_${S}_${T}_${TAG}.log
