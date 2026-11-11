#!/bin/bash
# v12: ONE commutator with three outcomes (copy / corrected sentence / [LEDGER] + draft), trained on commutation-verified
# targets whose candidate set includes the compressor's reasoning-mode repair; reasoning on demand; joint-verdict judging;
# recurrence test; scope ablation; program-aided baseline.  GPU 2 ONLY.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q19.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
fin() { log "$1"; touch $P/logs/Q19.done; exit 1; }
log start
$PY -c "import cocomem, run_corr, locomo, commutator, comm_eval2, comm_eval3, suffbench2, analyze, extras6, stat19" 2>/dev/null; \
$PY -c "import cocomem, run_corr, locomo, commutator, comm_eval2, comm_eval3, suffbench2, analyze" || fin PREFLIGHT_FAILED
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
find $P/results -name '*.tmp' -delete; find $P/results -name '*.lock' -delete
for p in 18431 18432 18433 18441 18442 18443; do bash stop_vllm.sh $p; done
# 1. targets: reasoning-mode global candidate + verification
if [ ! -e $P/models/tnet12/test.jsonl ]; then
  bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.88 >> $P/logs/q19.log
  up 18431 qwen3-8b || fin SERVER_DOWN_RELABEL
  $PY commutator.py relabel_cv2 --src $P/models/tnet10 --out $P/models/tnet12 >> $P/logs/q19_tnet.log 2>&1
fi
[ -e $P/models/tnet12/test.jsonl ] || fin RELABEL_FAILED
log "relabel $(grep -E '^(train|test) ' $P/logs/q19_tnet.log | tail -2 | tr '\n' ' ')"
bash stop_vllm.sh 18431
# 2. student: same recipe as every other student; three-outcome prompt
[ -e $P/models/tnet12/merged/config.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train --out $P/models/tnet12 --prompt 3 >> $P/logs/q19_tnet.log 2>&1
[ -e $P/models/tnet12/merged/config.json ] || fin TRAIN_FAILED
log trained
# 3. serving: compressor + the single commutator
bash start_srv.sh 2 18431 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.70 >> $P/logs/q19.log
bash start_tnet.sh $P/models/tnet12/merged 0.20 tnet12 18432 2 >> $P/logs/q19.log
up 18431 qwen3-8b && up 18432 tnet12 || fin SERVERS_DOWN
$PY commutator.py eval --out $P/models/tnet12 --served tnet12 --port 18432 --suffix _v12 --prompt 3 >> $P/logs/q19_tnet.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q19_tnet.log | tail -1 | cut -c1-200)"
# 4. wave 1: local, overwrite and ledger modes (LME + LoCoMo) with the benchmark
for s in comm12 comm12_overwrite comm12L; do rm -rf $P/results/main3/$s; $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 96 >> $P/logs/m3_$s.log 2>&1 & done
$PY suffbench2.py --units 150 --chain 60 --local tnet12 --out suffbench_v12.json > $P/logs/q19_suff.log 2>&1 &
for m in comm12 comm12L comm12_overwrite; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q19_$m.log 2>&1; done
wait
log "wave1 main=$(ls $P/results/main3/comm12 | grep -c 'json$'),$(ls $P/results/main3/comm12L | grep -c 'json$') loc=$(ls $P/results/locomo/comm12 | wc -l)"
[ "$(ls $P/results/main3/comm12 | grep -c 'json$')" -ge 200 ] || fin "RUNS_INCOMPLETE"
# 5. wave 2: reasoning on demand and the full-scope ablation; recurrence test
for s in comm12T comm12F; do rm -rf $P/results/main3/$s; $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 64 >> $P/logs/m3_$s.log 2>&1 & done
for m in comm12T; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q19_$m.log 2>&1; done
wait
log "wave2 main=$(ls $P/results/main3/comm12T | grep -c 'json$'),$(ls $P/results/main3/comm12F | grep -c 'json$')"
CM_PT=18432 $PY locomo.py run --regress --methods comm12,comm12T,ncf,coarse,rebuild,append >> $P/logs/locomo_reg.log 2>&1
log "regress $(ls $P/results/locomo_reg/comm12T 2>/dev/null | wc -l)"
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm12,comm12L,comm12T --out latency_g.json > $P/logs/q19_lat.log 2>&1
log latency
for p in 18431 18432; do bash stop_vllm.sh $p; done; sleep 5
# 6. judges, one after another on GPU 2: answers, then the joint-verdict direct test
G="$P/results/main3/comm12*/*.json,$P/results/locomo/comm12*/*.json,$P/results/locomo_reg/*/*.json"
EM=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm5,comm6,comm7,comm8,comm10,comm10_overwrite,comm11,comm12,comm12_overwrite,comm12L,comm12T,comm12F
for jm in "llama31 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ" "qwen25-7b /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ" "qwen3-8b /home/yaokang/models/Qwen3-8B-AWQ"; do
  set -- $jm
  bash start_gpu2.sh $2 $1 0.85 >> $P/logs/q19.log
  $PY judge.py --model $1 --glob "$G" >> $P/logs/q19_judge.log 2>&1
  $PY comm_eval3.py --model $1 --methods $EM >> $P/logs/q19_eval3.log 2>&1
  [ "$1" = llama31 ] && $PY comm_eval2.py --model llama31 --suffix _v12 --methods comm12,comm12_overwrite,comm12L,comm12T,comm12F >> $P/logs/q19_comm.log 2>&1
done
bash stop_vllm.sh 18431
log judged
# 7. tables and statistics
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm5,comm6,comm7,comm8,comm10,comm10_overwrite,comm11,comm12,comm12_overwrite,comm12L,comm12T,comm12F,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm12 --key label_$k --json $P/results/t19_main_$k.json > $P/results/t19_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm12 --key label_$k --valid_key label_qwen25-7b --json $P/results/t19f_main_$k.json > $P/results/t19f_main_$k.txt 2>&1
done
$PY analyze.py --tag main3 --methods $M --ref comm12 --key label_qwen25-7b --all --json $P/results/t19all_main_qwen25-7b.json > $P/results/t19all_main_qwen25-7b.txt 2>&1
L=comm12,comm12_overwrite,comm12L,comm12T,comm10,comm10_overwrite,comm11,comm5,comm6,comm7,comm8,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --ref comm12 --key label_$k --json $P/results/t19_locomo_$k.json > $P/results/t19_locomo_$k.txt 2>&1
done
$PY agg_eval3.py $EM > $P/results/eval3.txt 2>&1
$PY stat19.py comm12,comm12L,comm12T,comm10,comm11 ncf,coarse,rebuild,recompress,comm_zs,append > $P/results/stat19.txt 2>&1
$PY reg19.py comm12,comm12T,ncf,coarse,rebuild,append > $P/results/reg19.txt 2>&1
sed -i 's/^LME = "comm10,/LME = "comm12,comm12_overwrite,comm12L,comm12T,comm12F,comm10,/; s/^LOC = "comm10,/LOC = "comm12,comm12_overwrite,comm12L,comm12T,comm10,/; s/for m in ("comm10", /for m in ("comm12", "comm12L", "comm10", /' extras6.py
CM_EMB_DEVICE=cpu $PY extras6.py > $P/results/extras12.txt 2>&1; cp $P/results/extras6.json $P/results/extras12.json
[ -s $P/results/comm2_main3_v12.json ] && [ -s $P/results/eval3.json ] && log "ALL DONE" || log "EVAL_MISSING"
touch $P/logs/Q19.done
