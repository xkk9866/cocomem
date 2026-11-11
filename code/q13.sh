#!/bin/bash
# v8: well-formed information-level targets -> tnet7; full re-evaluation.  GPU 2 main, GPU 1 for tnet3.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q13.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
fin() { log "$1"; touch $P/logs/Q13.done; exit 1; }
log start
$PY -c "import cocomem, run_corr, locomo, commutator, comm_eval2, suffbench, analyze, extras6" || fin PREFLIGHT_FAILED
export CUDA_VISIBLE_DEVICES=2
for p in 18431 18432 18433; do bash stop_vllm.sh $p; done
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q13.log
[ -e $P/models/tnet7/test.jsonl ] || $PY commutator.py relabel --think --src $P/models/tnet2 --out $P/models/tnet7 >> $P/logs/q13_tnet.log 2>&1
[ -e $P/models/tnet7/test.jsonl ] || fin RELABEL_FAILED
log "relabel $(grep -E '^(train|test) ' $P/logs/q13_tnet.log | tail -2 | tr '\n' ' ')"
bash stop_vllm.sh 18431
[ -e $P/models/tnet7/merged/config.json ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py train --out $P/models/tnet7 >> $P/logs/q13_tnet.log 2>&1
[ -e $P/models/tnet7/merged/config.json ] || fin TRAIN_FAILED
log trained
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.70 >> $P/logs/q13.log
bash start_tnet.sh $P/models/tnet7/merged 0.20 tnet7 18432 >> $P/logs/q13.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 18433 1 >> $P/logs/q13.log
up 18431 qwen3-8b && up 18432 tnet7 && up 18433 tnet3 || fin SERVERS_DOWN
$PY commutator.py eval --out $P/models/tnet7 --served tnet7 --port 18432 --suffix _v9 >> $P/logs/q13_tnet.log 2>&1
$PY commutator.py eval --out $P/models/tnet7 --served qwen3-8b --port 18431 --suffix _v9 >> $P/logs/q13_tnet.log 2>&1
log "eval $(grep -E '^\{' $P/logs/q13_tnet.log | tail -2 | cut -c1-100 | tr '\n' ' ')"
export CM_EMB_DEVICE=cpu
for s in comm7 comm7_overwrite comm8; do rm -rf $P/results/main3/$s; $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 96 >> $P/logs/m3_$s.log 2>&1 & done
for m in comm7 comm7_overwrite comm8; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q13_$m.log 2>&1 & done
$PY suffbench.py --units 150 --chain 60 --local tnet7 --out suffbench_v9.json > $P/logs/q13_suff.log 2>&1 &
wait
R="main=$(ls $P/results/main3/comm7 | wc -l),$(ls $P/results/main3/comm8 | wc -l) loc=$(ls $P/results/locomo/comm7 | wc -l),$(ls $P/results/locomo/comm8 | wc -l)"
log "runs $R"
[ "$(ls $P/results/main3/comm7 | wc -l)" -ge 200 ] || fin "RUNS_INCOMPLETE $R"
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm7,comm8 --out latency_e.json > $P/logs/q13_lat.log 2>&1
log latency
for p in 18431 18432 18433; do bash stop_vllm.sh $p; done
G="$P/results/main3/comm7*/*.json,$P/results/main3/comm8/*.json,$P/results/locomo/comm7*/*.json,$P/results/locomo/comm8/*.json"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q13.log
$PY judge.py --model llama31 --glob "$G" > $P/logs/q13_judge.log 2>&1
$PY comm_eval2.py --model llama31 --suffix _v9 --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,comm5,comm6,comm7,comm8,recompress,rebuild >> $P/logs/q13_comm.log 2>&1
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q13.log
$PY judge.py --model qwen25-7b --glob "$G" >> $P/logs/q13_judge.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q13.log
$PY judge.py --model qwen3-8b --glob "$G" >> $P/logs/q13_judge.log 2>&1
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,comm5,comm6,comm7,comm7_overwrite,comm8,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm7 --key label_$k --json $P/results/t13_main_$k.json > $P/results/t13_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm7 --key label_$k --valid_key label_qwen25-7b --json $P/results/t13f_main_$k.json > $P/results/t13f_main_$k.txt 2>&1
done
L=comm7,comm7_overwrite,comm8,comm5,comm6,comm4,comm3,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t13_locomo_$k.json > $P/results/t13_locomo_$k.txt 2>&1
done
sed -i 's/^LME = "comm5,comm6,/LME = "comm7,comm8,comm5,comm6,/; s/^LOC = "comm5,/LOC = "comm7,comm7_overwrite,comm8,comm5,/; s/for m in ("comm5",/for m in ("comm7", "comm8", "comm5",/' extras6.py
$PY extras6.py > $P/results/extras9.txt 2>&1; cp $P/results/extras6.json $P/results/extras9.json
log "ALL DONE"; touch $P/logs/Q13.done
