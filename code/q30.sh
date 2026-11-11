#!/bin/bash
# q30: CoCoMem v20 = derivation witnesses + witness-conditioned commutator (one network, one round), single seed.
# GPUs 6 and 7 only. Two teacher replicas, parallel student training, separate inference GPUs.
#   P1 witness pass (write time)            -> states/witness/, states/witness_locomo/
#   P2 verified witness-aware targets       -> models/w20/{train,test}.jsonl ; variants models/tnet20{,nw,nv,nr}
#   P3 four students (Qwen3-4B LoRA)        -> models/tnet20*/merged
#   P4 runs with tnet20 + tnet20nw          -> results/main3/comm20*, results/locomo/comm20*, suffbench_v20.json, latency_v20.json
#   P5 runs with tnet20nv + tnet20nr        -> results/main3/comm20_nv|nr, results/locomo/comm20_nr
#   P6 three judges (+ official baselines), direct test, coverage
#   P7 tables / statistics                  -> results/t30_*.json, stat30*.json, xtab.json, extras30.json; r30.tgz
set -Eeuo pipefail
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
L=$P/logs/q30.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $L; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
fin() { log "$1"; echo "$1" > $P/logs/Q30.failed; exit 1; }
trap 'fin "FAILED line=$LINENO status=$?"' ERR
cnt() { find "$1" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l || true; }
export CUDA_VISIBLE_DEVICES=6 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec 9>"$P/logs/q30.run.lock"
flock -n 9 || { echo 'q30 is already running'; exit 1; }
Q8=/home/yaokang/models/Qwen3-8B-AWQ
rm -f $P/logs/Q30.done $P/logs/Q30.failed
log 'start GPUs=6,7 seed=0'
$PY -m py_compile cocomem.py run_corr.py locomo.py commutator.py witness.py suffbench4.py comm_eval2.py coverage.py analyze.py extras6.py stat19.py reg19.py || fin PREFLIGHT_FAILED
if [ "${Q30_RESUME_EVAL:-0}" != 1 ]; then
for p in 18431 18432 18433 18435; do bash stop_vllm.sh $p & done
wait

# ---------------------------------------------------------------- P1 + P2: compressor only (0.85)
mkdir -p $P/states/witness $P/states/witness_locomo
NW=$(cnt $P/states/witness); NS=$($PY witness.py count --kind lme --required-only)
if [ "$NW" -lt "$NS" ] || [ ! -e $P/models/w20/test.jsonl ]; then
  bash start_srv.sh 6 18431 $Q8 qwen3-8b 0.85 >> $L & p6=$!
  bash start_srv.sh 7 18435 $Q8 qwen3-8b 0.85 >> $L & p7=$!
  wait "$p6"; wait "$p7"
  up 18431 qwen3-8b && up 18435 qwen3-8b || fin SERVER_DOWN_P1
  export CM_Q8_REPLICAS=18431,18435
  $PY witness.py annotate --kind lme --required-only --conc 192 >> $P/logs/q30_witness.log 2>&1 & wl=$!
  $PY witness.py annotate --kind locomo --conc 48 >> $P/logs/q30_witness_locomo.log 2>&1 & wc=$!
  wait "$wl"; wait "$wc"
  log "witness lme=$(ls $P/states/witness | wc -l)/$NS locomo=$(ls $P/states/witness_locomo | wc -l)"
  mkdir -p $P/models/w20
  [ -e $P/models/w20/test.jsonl ] || $PY commutator.py relabel_w --src $P/models/tnet10 --out $P/models/w20 --conc 192 >> $P/logs/q30_tnet.log 2>&1
fi
[ -e $P/models/w20/test.jsonl ] || fin RELABEL_FAILED
log "relabel $(grep -E '^(train|test) ' $P/logs/q30_tnet.log | tail -2 | tr '\n' ' ')"
[ -e $P/models/tnet20nr/test.jsonl ] || $PY commutator.py data_w --src $P/models/w20 >> $P/logs/q30_tnet.log 2>&1
unset CM_Q8_REPLICAS
bash stop_vllm.sh 18431 & sp6=$!
bash stop_vllm.sh 18435 & sp7=$!
wait "$sp6"; wait "$sp7"

# ---------------------------------------------------------------- P3: four students, same recipe (r16, 1 epoch)
train_one() {
  local g=$1 v=$2
  if [ ! -e $P/models/$v/train_meta.json ]; then
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY commutator.py train --out $P/models/$v --max_len 3072 --tok_budget 4096 >> $P/logs/q30_train_$v.log 2>&1
  fi
  [ -e $P/models/$v/merged/config.json ] && [ -e $P/models/$v/train_meta.json ] || fin "TRAIN_FAILED_$v"
  log "trained $v GPU=$g"
}
( train_one 6 tnet20; train_one 6 tnet20nv ) & tr6=$!
( train_one 7 tnet20nw; train_one 7 tnet20nr ) & tr7=$!
wait "$tr6"; wait "$tr7"
fi
for v in tnet20 tnet20nw tnet20nv tnet20nr; do
  [ -s "$P/models/$v/train_meta.json" ] && [ -s "$P/models/$v/merged/config.json" ] || fin "MISSING_MODEL_$v"
done

# ---------------------------------------------------------------- P4: 8B (0.50) + tnet20 (18432) + tnet20nw (18433)
serve4() {   # $1 name@18432  $2 name@18433
  if ! up 18431 qwen3-8b; then
    bash start_srv.sh 6 18431 $Q8 qwen3-8b 0.85 >> $L || return 1
  fi
  bash start_tnet.sh $P/models/$1/merged 0.40 $1 18432 7 >> $L || return 1
  bash start_tnet.sh $P/models/$2/merged 0.40 $2 18433 7 >> $L || return 1
  up 18431 qwen3-8b && up 18432 $1 && up 18433 $2
}
serve4 tnet20 tnet20nw || fin SERVERS_DOWN_P4
for v in tnet20:18432 tnet20nw:18433; do
  n=${v%%:*}; pt=${v##*:}
  [ -e $P/results/tnet_eval_${n}_v20.json ] || $PY commutator.py eval --out $P/models/$n --served $n --port $pt --suffix _v20 >> $P/logs/q30_tnet.log 2>&1
done
$PY witness.py stats --required-only --pairs $P/models/tnet20/test.jsonl --out witness_stats.json > $P/logs/q30_wstats.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q30_tnet.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
# LongMemEval-Corr (201 test questions): main method + ablations that use tnet20 / tnet20nw
CM_PT=18432 $PY run_corr.py --splits test200 --tag main3 --strategies comm20,comm20_cite,comm20_overwrite --inst 8 --conc 64 >> $P/logs/m3_comm20.log 2>&1 & main_left=$!
CM_PT=18433 $PY run_corr.py --splits test200 --tag main3 --strategies comm20_nw --inst 6 --conc 64 >> $P/logs/m3_comm20_nw.log 2>&1 & main_right=$!
wait "$main_left"; wait "$main_right"
CM_PT=18432 $PY run_corr.py --splits test200 --tag main3 --strategies comm20_zs --inst 6 --conc 64 >> $P/logs/m3_comm20_zs.log 2>&1 & zs_job=$!
# LoCoMo-Corr (sequential corrections) + recurrence (re-ask after every later correction)
( CM_PT=18432 $PY locomo.py run --methods comm20 >> $P/logs/locomo_q30.log 2>&1
  CM_PT=18432 $PY locomo.py run --regress --methods comm20 >> $P/logs/locomo_q30.log 2>&1 ) & loc_main=$!
CM_PT=18433 $PY locomo.py run --methods comm20_nw >> $P/logs/locomo_q30_nw.log 2>&1 & loc_nw=$!
wait "$zs_job"; wait "$loc_main"; wait "$loc_nw"
log "P4 main=$(cnt $P/results/main3/comm20),$(cnt $P/results/main3/comm20_nw),$(cnt $P/results/main3/comm20_cite),$(cnt $P/results/main3/comm20_overwrite),$(cnt $P/results/main3/comm20_zs) loc=$(cnt $P/results/locomo/comm20),$(cnt $P/results/locomo/comm20_nw) reg=$(cnt $P/results/locomo_reg/comm20)"
[ "$(cnt $P/results/main3/comm20)" -ge 195 ] || fin "RUNS_INCOMPLETE_P4"
# programmatic benchmark: single corrections + 100-step chains with retractions + witness recall
if [ ! -e $P/logs/q30_suff_a.done ]; then
$PY suffbench4.py --students tnet20:18432,tnet20nw:18433 --systems copy,comm20,comm20_nw,teacher_w,ledger_all,think_all --tag a > $P/logs/q30_suff_a.log 2>&1
  touch $P/logs/q30_suff_a.done
fi
log "suffbench a $(ls -la $P/results/suffbench_v20.json 2>&1 | cut -c1-80)"
CM_NOCACHE=1 CM_PT=18432 $PY bench_latency.py --n 30 --methods comm20,ncf,coarse,recompress,rebuild --out latency_v20.json > $P/logs/q30_lat.log 2>&1
log latency

# ---------------------------------------------------------------- P5: tnet20nv (18432) + tnet20nr (18433)
serve4 tnet20nv tnet20nr || fin SERVERS_DOWN_P5
for v in tnet20nv:18432 tnet20nr:18433; do
  n=${v%%:*}; pt=${v##*:}
  [ -e $P/results/tnet_eval_${n}_v20.json ] || $PY commutator.py eval --out $P/models/$n --served $n --port $pt --suffix _v20 >> $P/logs/q30_tnet.log 2>&1
done
CM_PT=18432 $PY run_corr.py --splits test200 --tag main3 --strategies comm20_nv --inst 6 --conc 64 >> $P/logs/m3_comm20_nv.log 2>&1 & nv_job=$!
CM_PT=18433 $PY run_corr.py --splits test200 --tag main3 --strategies comm20_nr --inst 6 --conc 64 >> $P/logs/m3_comm20_nr.log 2>&1 & nr_job=$!
CM_PT=18433 $PY locomo.py run --methods comm20_nr >> $P/logs/locomo_q30_nr.log 2>&1 & loc_nr=$!
wait "$nv_job"; wait "$nr_job"; wait "$loc_nr"
if [ ! -e $P/logs/q30_suff_b.done ]; then
$PY suffbench4.py --students tnet20nv:18432,tnet20nr:18433 --systems comm20_nv,comm20_nr --tag b > $P/logs/q30_suff_b.log 2>&1
  touch $P/logs/q30_suff_b.done
fi
log "P5 main=$(cnt $P/results/main3/comm20_nv),$(cnt $P/results/main3/comm20_nr) loc=$(cnt $P/results/locomo/comm20_nr)"
for p in 18431 18432 18433; do bash stop_vllm.sh $p & done; wait

# ---------------------------------------------------------------- P6: judges, direct test, coverage
G="$P/results/main3/comm20*/*.json,$P/results/locomo/comm20*/*.json,$P/results/locomo_reg/comm20/*.json,$P/results/main3/off_*/*.json,$P/results/locomo/off_*/*.json,$P/results/locomo_reg/off_*/*.json"
(
  bash start_srv.sh 6 18431 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.85 >> $L
  $PY judge.py --model qwen25-7b --port 18431 --glob "$G" >> $P/logs/q30_judge_q25.log 2>&1
  bash stop_vllm.sh 18431
  bash start_srv.sh 6 18431 $Q8 qwen3-8b 0.85 >> $L
  $PY judge.py --model qwen3-8b --port 18431 --glob "$G" >> $P/logs/q30_judge_q3.log 2>&1
  $PY coverage.py --port 18431 --conc 96 --suffix _v20 --methods orig,append,slots,simupdate,coarse,ncf,recompress,rebuild,comm10,comm20,comm20_nw,comm20_cite,comm20_nv,comm20_nr,comm20_zs >> $P/logs/q30_cov.log 2>&1
) & j6=$!
(
  bash start_srv.sh 7 18435 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.85 >> $L
  $PY judge.py --model llama31 --port 18435 --glob "$G" >> $P/logs/q30_judge_llama.log 2>&1
  $PY comm_eval2.py --model llama31 --port 18435 --suffix _v20 --methods comm20,comm20_nw,comm20_cite,comm20_nv,comm20_nr,comm20_overwrite,comm20_zs >> $P/logs/q30_comm.log 2>&1
) & j7=$!
wait "$j6"; wait "$j7"
bash stop_vllm.sh 18431 & e6=$!
bash stop_vllm.sh 18435 & e7=$!
wait "$e6"; wait "$e7"
log judged

# ---------------------------------------------------------------- P7: tables and statistics
M=append,slots,simupdate,coarse,ncf,spec,recompress,rebuild,comm10,comm20,comm20_nw,comm20_cite,comm20_nv,comm20_nr,comm20_overwrite,comm20_zs,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm20 --key label_$k --json $P/results/t30_main_$k.json > $P/results/t30_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm20 --key label_$k --valid_key label_qwen25-7b --json $P/results/t30f_main_$k.json > $P/results/t30f_main_$k.txt 2>&1
done
$PY analyze.py --tag main3 --methods $M --ref comm20 --key label_qwen25-7b --all --json $P/results/t30all_main_qwen25-7b.json > $P/results/t30all_main_qwen25-7b.txt 2>&1
# official baselines finished only part of LongMemEval: one block per baseline on the questions it finished
for b in off_mem0 off_amem; do
  Q=$(ls $P/results/main3/$b | grep 'json$' | sed 's/.json$//' | paste -sd,)
  $PY analyze.py --tag main3 --methods rawrag,membank,$b,ncf,rebuild,comm20,oracle --ref comm20 --key label_qwen25-7b --qids "$Q" --json $P/results/t30${b#off_}_main_qwen25-7b.json > $P/results/t30${b#off_}_main_qwen25-7b.txt 2>&1
done
LO=append,slots,simupdate,coarse,ncf,recompress,rebuild,comm10,comm20,comm20_nw,comm20_nr,rawrag,membank,mem0,amem,off_mem0,off_amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $LO --ref comm20 --key label_$k --json $P/results/t30_locomo_$k.json > $P/results/t30_locomo_$k.txt 2>&1
done
$PY stat19.py comm20 rawrag,membank,mem0,simupdate,coarse,ncf,recompress,rebuild,comm10,comm20_nw,comm20_cite,comm20_nv,comm20_nr,comm20_zs,append > $P/results/stat30.txt 2>&1; cp $P/results/stat19.json $P/results/stat30.json
$PY reg19.py comm20,ncf,coarse,rebuild,append > $P/results/reg30.txt 2>&1; cp $P/results/reg19.json $P/results/reg30.json
python3 $P/xtab.py > $P/results/xtab.txt 2>&1
X_LME=comm20,comm10,append,slots,simupdate,coarse,ncf,recompress,spec,rebuild X_LOC=comm20,comm10,ncf,coarse,recompress,rebuild,append,rawrag,membank,mem0,amem X_SEQ=comm20,comm10,ncf,coarse,recompress,rebuild X_OUT=extras30.json $PY extras6.py > $P/results/extras30.txt 2>&1
cp $P/models/w20/label_cost.json $P/results/label_cost_v20.json
for v in tnet20 tnet20nw tnet20nv tnet20nr; do cp $P/models/$v/train_meta.json $P/results/train_meta_$v.json; done
cd $P/results
shopt -s nullglob
tar czf $P/r30.tgz t30*.json stat30.json reg30.json xtab.json extras30.json suffbench_v20.json latency_v20.json witness_stats.json comm2_main3_v20.json coverage_main3_v20.json tnet_eval_tnet20*_v20.json judgecal_*.json label_cost_v20.json train_meta_tnet20*.json
cd $P/code
[ -s $P/results/t30_main_qwen25-7b.json ] && [ -s $P/results/suffbench_v20.json ] && [ -s $P/results/comm2_main3_v20.json ] || fin OUTPUT_MISSING
log "ALL DONE"
touch $P/logs/Q30.done
