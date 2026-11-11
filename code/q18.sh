#!/bin/bash
# v10: commutation-verified distillation -> tnet10; full re-evaluation.  GPU 2 ONLY.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q18.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
fin() { log "$1"; touch $P/logs/Q18.done; exit 1; }
log start
$PY -c "import cocomem, run_corr, locomo, commutator, comm_eval2, suffbench, analyze, extras6" || fin PREFLIGHT_FAILED
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
for p in 18431 18432 18433 18441 18442 18443; do bash stop_vllm.sh $p; done
# 1. targets
if [ ! -e $P/models/tnet10/test.jsonl ]; then
  bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.85 >> $P/logs/q18.log
  up 18431 qwen3-8b || fin SERVER_DOWN_RELABEL
  $PY commutator.py relabel_cv --out $P/models/tnet10 >> $P/logs/q18_tnet.log 2>&1
fi
[ -e $P/models/tnet10/test.jsonl ] || fin RELABEL_FAILED
log "relabel $(grep -E '^(train|test) ' $P/logs/q18_tnet.log | tail -2 | tr '\n' ' ')"
bash stop_vllm.sh 18431
# 2. student (same recipe as tnet5/tnet7: Qwen3-4B LoRA r16, 1 epoch)
[ -e $P/models/tnet10/merged/config.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train --out $P/models/tnet10 >> $P/logs/q18_tnet.log 2>&1
[ -e $P/models/tnet10/merged/config.json ] || fin TRAIN_FAILED
log trained
# 3. runs: compressor + student + sufficiency net, all on GPU 2
bash start_srv.sh 2 18431 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.55 >> $P/logs/q18.log
bash start_tnet.sh $P/models/tnet10/merged 0.17 tnet10 18432 2 >> $P/logs/q18.log
bash start_tnet.sh $P/models/tnet3/merged 0.17 tnet3 18433 2 >> $P/logs/q18.log
up 18431 qwen3-8b && up 18432 tnet10 && up 18433 tnet3 || fin SERVERS_DOWN
$PY commutator.py eval --out $P/models/tnet10 --served tnet10 --port 18432 --suffix _v10 >> $P/logs/q18_tnet.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q18_tnet.log | tail -1 | cut -c1-160)"
for s in comm10 comm10_overwrite comm11; do rm -rf $P/results/main3/$s; $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 96 >> $P/logs/m3_$s.log 2>&1 & done
for m in comm10 comm10_overwrite comm11; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q18_$m.log 2>&1 & done
$PY suffbench.py --units 150 --chain 60 --local tnet10 --out suffbench_v10.json > $P/logs/q18_suff.log 2>&1 &
wait
R="main=$(ls $P/results/main3/comm10 | grep -c 'json$'),$(ls $P/results/main3/comm11 | grep -c 'json$') loc=$(ls $P/results/locomo/comm10 | wc -l),$(ls $P/results/locomo/comm11 | wc -l)"
log "runs $R"
[ "$(ls $P/results/main3/comm10 | grep -c 'json$')" -ge 200 ] || fin "RUNS_INCOMPLETE $R"
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm10,comm11 --out latency_f.json > $P/logs/q18_lat.log 2>&1
log latency
for p in 18431 18432 18433; do bash stop_vllm.sh $p; done
bash stop_vllm.sh 18431; sleep 5
# 4. three judges, one after another on GPU 2
G="$P/results/main3/comm10*/*.json,$P/results/main3/comm11/*.json,$P/results/locomo/comm10*/*.json,$P/results/locomo/comm11/*.json"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.85 >> $P/logs/q18.log
$PY judge.py --model llama31 --glob "$G" > $P/logs/q18_judge.log 2>&1
$PY comm_eval2.py --model llama31 --suffix _v10 --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,comm5,comm6,comm7,comm8,comm10,comm11,comm10_overwrite,recompress,rebuild >> $P/logs/q18_comm.log 2>&1
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.85 >> $P/logs/q18.log
$PY judge.py --model qwen25-7b --glob "$G" >> $P/logs/q18_judge.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.85 >> $P/logs/q18.log
$PY judge.py --model qwen3-8b --glob "$G" >> $P/logs/q18_judge.log 2>&1
bash stop_vllm.sh 18431
log judged
# 5. tables
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm5,comm5_overwrite,comm6,comm7,comm7_overwrite,comm8,comm10,comm10_overwrite,comm11,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm10 --key label_$k --json $P/results/t18_main_$k.json > $P/results/t18_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm10 --key label_$k --valid_key label_qwen25-7b --json $P/results/t18f_main_$k.json > $P/results/t18f_main_$k.txt 2>&1
done
L=comm10,comm10_overwrite,comm11,comm7,comm7_overwrite,comm8,comm5,comm6,comm4,comm3,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --ref comm10 --key label_$k --json $P/results/t18_locomo_$k.json > $P/results/t18_locomo_$k.txt 2>&1
done
$PY stat18.py > $P/results/stat18.txt 2>&1
$PY extras6.py > $P/results/extras10.txt 2>&1; cp $P/results/extras6.json $P/results/extras10.json
[ -s $P/results/comm2_main3_comm10.json ] && log "ALL DONE" || log "COMM_EVAL_MISSING"
touch $P/logs/Q18.done
