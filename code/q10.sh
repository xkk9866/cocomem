#!/bin/bash
# v7: verified commutator (comm4) on LongMemEval-Corr and LoCoMo-Corr, verifiable sufficiency benchmark,
# no-cache latency, three judges, fixed-set re-analysis, direct commutativity test with semantic equivalence.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q10.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
log start
$PY -c "import cocomem, run_corr, locomo, commutator, comm_eval2, suffbench, analyze, extras6" || { log PREFLIGHT_FAILED; touch $P/logs/Q10.done; exit 1; }
export CUDA_VISIBLE_DEVICES=2
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.70 >> $P/logs/q10.log
bash start_tnet.sh $P/models/tnet2/merged 0.20 tnet2 18432 >> $P/logs/q10.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 18433 1 >> $P/logs/q10.log
up 18431 qwen3-8b && up 18432 tnet2 && up 18433 tnet3 || { log SERVERS_DOWN; touch $P/logs/Q10.done; exit 1; }
export CM_EMB_DEVICE=cpu
for s in comm4 comm4_overwrite; do rm -rf $P/results/main3/$s; $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 96 >> $P/logs/m3_$s.log 2>&1 & done
for m in comm4 comm4_overwrite; do rm -rf $P/results/locomo/$m; $PY locomo.py run --methods $m >> $P/logs/locomo_q10_$m.log 2>&1 & done
$PY suffbench.py --units 150 --chain 60 > $P/logs/q10_suff.log 2>&1 &
wait
log "runs main=$(ls $P/results/main3/comm4 | wc -l) ow=$(ls $P/results/main3/comm4_overwrite | wc -l) loc=$(ls $P/results/locomo/comm4 | wc -l) suff=$(test -s $P/results/suffbench.json && echo y)"
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm4 --out latency_c.json > $P/logs/q10_lat.log 2>&1
log latency
bash stop_vllm.sh 18432; bash stop_vllm.sh 18433; bash stop_vllm.sh 18431
G="$P/results/main3/comm4*/*.json,$P/results/locomo/comm4*/*.json"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q10.log
$PY judge.py --model llama31 --glob "$G" > $P/logs/q10_judge.log 2>&1
$PY comm_eval2.py --model llama31 --suffix _v7 --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,recompress,rebuild >> $P/logs/q10_comm.log 2>&1
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q10.log
$PY judge.py --model qwen25-7b --glob "$G" >> $P/logs/q10_judge.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q10.log
$PY judge.py --model qwen3-8b --glob "$G" >> $P/logs/q10_judge.log 2>&1
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm_overwrite,comm2,comm2_overwrite,comm3,comm3_nofallback,comm3_overwrite,comm4,comm4_overwrite,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm4 --key label_$k --json $P/results/t10_main_$k.json > $P/results/t10_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm4 --key label_$k --valid_key label_qwen25-7b --json $P/results/t10f_main_$k.json > $P/results/t10f_main_$k.txt 2>&1
done
L=comm4,comm4_overwrite,comm3,comm3_nofallback,comm3_overwrite,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t10_locomo_$k.json > $P/results/t10_locomo_$k.txt 2>&1
done
$PY extras6.py > $P/results/extras7.txt 2>&1; cp $P/results/extras6.json $P/results/extras7.json
log "ALL DONE"; touch $P/logs/Q10.done
