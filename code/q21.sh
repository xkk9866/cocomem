#!/bin/bash
# q21: answers to the 4th review.  GPU 2 ONLY.
#   0. direct-test additions: information-completeness (Missing / Equivalence), deletion adversaries, judge calibration on computed truth
#   1. v13: counterfactual aggregate curriculum -> verified targets (incl. reasoning-mode repair) -> ONE three-outcome network
#   2. evaluation of v13 (programmatic benchmark, LME-Corr, LoCoMo, recurrence, latency), judges, statistics
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q21.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
fin() { log "$1"; touch $P/logs/Q21.done; exit 1; }
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
log start
find $P/results -name '*.tmp' -delete; find $P/results -name '*.lock' -delete
for p in 18431 18432; do bash stop_vllm.sh $p; done
# ---- 0. direct-test additions
$PY mk_delete.py >> $P/logs/q21_misc.log 2>&1
$PY stat19.py comm10,comm11 ncf,coarse,rebuild,recompress,comm_zs,append,comm10 label_llama31 > /dev/null 2>&1
$PY stat19.py comm10,comm11 ncf,coarse,rebuild,recompress,comm_zs,append,comm10 label_qwen3-8b > /dev/null 2>&1
log "stat19 other judges $(ls $P/results/stat19_*.json | tr '\n' ' ')"
for jm in "llama31 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ" "qwen25-7b /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ" "qwen3-8b /home/yaokang/models/Qwen3-8B-AWQ"; do
  set -- $jm
  bash start_gpu2.sh $2 $1 0.85 >> $P/logs/q21.log
  up 18431 $1 || fin "JUDGE_DOWN_$1"
  $PY judgecal.py --model $1 >> $P/logs/q21_cal.log 2>&1
  [ "$1" = llama31 ] && $PY comm_eval2.py --model llama31 --suffix _del --methods del_oracle,del_comm10 >> $P/logs/q21_comm.log 2>&1
done
bash stop_vllm.sh 18431
python3 $P/xtab.py > $P/results/xtab.txt 2>&1
log "phase0 done: $(ls $P/results/judgecal_*.json | wc -l) calibrations"
# ---- 1. v13 curriculum
if [ ! -e $P/models/synth_out/train.jsonl ]; then
  bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.88 >> $P/logs/q21.log
  up 18431 qwen3-8b || fin SERVER_DOWN_SYNTH
  [ -e $P/models/synth/train.jsonl ] || $PY synth_cur.py >> $P/logs/q21_synth.log 2>&1
  [ -e $P/models/synth/train.jsonl ] || fin SYNTH_FAILED
  log "synth $(tail -1 $P/logs/q21_synth.log)"
  mkdir -p $P/models/synth_out
  $PY commutator.py relabel_cv2 --src $P/models/synth --out $P/models/synth_out >> $P/logs/q21_tnet.log 2>&1
fi
[ -e $P/models/synth_out/train.jsonl ] || fin RELABEL_FAILED
log "relabel $(grep -E '^(train|test) ' $P/logs/q21_tnet.log | tail -2 | tr '\n' ' ')"
bash stop_vllm.sh 18431
mkdir -p $P/models/tnet13
cat $P/models/tnet12/train.jsonl $P/models/synth_out/train.jsonl > $P/models/tnet13/train.jsonl
cp $P/models/tnet12/test.jsonl $P/models/tnet13/test.jsonl
[ -e $P/models/tnet13/merged/config.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train --out $P/models/tnet13 --prompt 3 >> $P/logs/q21_tnet.log 2>&1
[ -e $P/models/tnet13/merged/config.json ] || fin TRAIN_FAILED
log trained
bash start_srv.sh 2 18431 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.70 >> $P/logs/q21.log
bash start_tnet.sh $P/models/tnet13/merged 0.20 tnet13 18432 2 >> $P/logs/q21.log
up 18431 qwen3-8b && up 18432 tnet13 || fin SERVERS_DOWN
$PY commutator.py eval --out $P/models/tnet13 --served tnet13 --port 18432 --suffix _v13 --prompt 3 >> $P/logs/q21_tnet.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q21_tnet.log | tail -1 | cut -c1-300)"
$PY suffbench2.py --units 150 --chain 60 --local tnet13 --out suffbench_v13.json > $P/logs/q21_suff.log 2>&1
log "suffbench done $(ls -la $P/results/suffbench_v13.json | cut -c1-60)"
# ---- 2. LME-Corr, LoCoMo, recurrence, latency
for s in comm13 comm13L comm13T; do rm -rf $P/results/main3/$s; $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 64 >> $P/logs/m3_$s.log 2>&1 & done
for m in comm13 comm13L comm13T; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q21_$m.log 2>&1; done
wait
log "wave main=$(ls $P/results/main3/comm13 | grep -c 'json$'),$(ls $P/results/main3/comm13L | grep -c 'json$'),$(ls $P/results/main3/comm13T | grep -c 'json$') loc=$(ls $P/results/locomo/comm13 | wc -l)"
CM_PT=18432 $PY locomo.py run --regress --methods comm13 >> $P/logs/locomo_reg3.log 2>&1
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm13,comm13L,comm13T --out latency_h.json > $P/logs/q21_lat.log 2>&1
log latency
for p in 18431 18432; do bash stop_vllm.sh $p; done; sleep 5
# ---- judges: answers (3 judges), then the direct test (Llama, last so that comm2 files are final)
G="$P/results/main3/comm13*/*.json,$P/results/locomo/comm13*/*.json,$P/results/locomo_reg/comm13/*.json"
for jm in "qwen25-7b /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ" "qwen3-8b /home/yaokang/models/Qwen3-8B-AWQ" "llama31 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ"; do
  set -- $jm
  bash start_gpu2.sh $2 $1 0.85 >> $P/logs/q21.log
  $PY judge.py --model $1 --glob "$G" >> $P/logs/q21_judge.log 2>&1
  [ "$1" = llama31 ] && $PY comm_eval2.py --model llama31 --suffix _v13 --methods comm13,comm13L,comm13T >> $P/logs/q21_comm.log 2>&1
done
bash stop_vllm.sh 18431
log judged
# ---- tables and statistics
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm5,comm6,comm7,comm8,comm10,comm10_overwrite,comm11,comm12,comm12L,comm12T,comm13,comm13L,comm13T,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm10 --key label_$k --json $P/results/t21_main_$k.json > $P/results/t21_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm10 --key label_$k --valid_key label_qwen25-7b --json $P/results/t21f_main_$k.json > $P/results/t21f_main_$k.txt 2>&1
done
$PY analyze.py --tag main3 --methods $M --ref comm10 --key label_qwen25-7b --all --json $P/results/t21all_main_qwen25-7b.json > $P/results/t21all_main_qwen25-7b.txt 2>&1
L=comm13,comm13L,comm13T,comm12,comm12L,comm12T,comm10,comm11,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --ref comm10 --key label_$k --json $P/results/t21_locomo_$k.json > $P/results/t21_locomo_$k.txt 2>&1
done
python3 $P/xtab.py > $P/results/xtab.txt 2>&1
$PY stat19.py comm10,comm11,comm13,comm13L,comm13T rawrag,ncf,coarse,rebuild,recompress,append,comm_zs,comm10,comm11 > $P/results/stat21.txt 2>&1; cp $P/results/stat19.json $P/results/stat21.json
$PY reg19.py comm12,comm13,ncf,coarse,rebuild,append > $P/results/reg21.txt 2>&1
sed -i 's/^LME = "comm12,/LME = "comm13,comm13L,comm13T,comm12,/; s/^LOC = "comm12,/LOC = "comm13,comm13L,comm13T,comm12,/; s/for m in ("comm12", /for m in ("comm13", "comm13L", "comm12", /' extras6.py
CM_EMB_DEVICE=cpu $PY extras6.py > $P/results/extras13.txt 2>&1; cp $P/results/extras6.json $P/results/extras13.json
cd $P/results && tar czf $P/r21.tgz judgecal_*.json xtab.json stat19*.json stat21.json reg19.json suffbench_v13.json latency_h.json extras13.json comm2_main3_v13.json comm2_main3_del.json t21*.json tnet_eval_tnet13_v13.json 2>/dev/null
cp $P/models/tnet13/test.jsonl $P/tnet13_test.jsonl 2>/dev/null
[ -s $P/results/suffbench_v13.json ] && [ -s $P/results/t21_main_qwen25-7b.json ] && log "ALL DONE" || log "OUTPUT_MISSING"
touch $P/logs/Q21.done
