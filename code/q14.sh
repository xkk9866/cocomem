#!/bin/bash
# Parallel helpers for q13: GPU0 = second 8B + tnet replicas working the same queue from the other end;
# after q13's main runs, the three judges run concurrently on GPU0/1/2.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q14.log; }
log start
bash start_srv.sh 0 18451 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.50 >> $P/logs/q14.log
bash start_tnet.sh $P/models/tnet7/merged 0.20 tnet7 18452 0 >> $P/logs/q14.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 18453 0 >> $P/logs/q14.log
export CM_P8=18451 CM_PT=18452 CM_PS=18453 CM_REVERSE=1 CM_EMB_DEVICE=cpu
$PY run_corr.py --splits test200 --tag main3 --strategies comm7 --inst 10 --conc 96 >> $P/logs/m3_comm7_r.log 2>&1 &
$PY run_corr.py --splits test200 --tag main3 --strategies comm7_overwrite --inst 10 --conc 96 >> $P/logs/m3_comm7o_r.log 2>&1 &
$PY run_corr.py --splits test200 --tag main3 --strategies comm8 --inst 10 --conc 96 >> $P/logs/m3_comm8_r.log 2>&1 &
$PY locomo.py run --methods comm7 >> $P/logs/locomo_r7.log 2>&1 &
$PY locomo.py run --methods comm8 >> $P/logs/locomo_r8.log 2>&1 &
wait
log "helpers done"
unset CM_P8 CM_PT CM_PS CM_REVERSE
for p in 18451 18452 18453; do bash stop_vllm.sh $p; done
# wait for q13's own runs and no-cache latency, then take over the judging stage
while ! grep -q ' latency$' $P/logs/q13.log; do sleep 60; done
kill $(pgrep -f '[q]13.sh') 2>/dev/null; sleep 3
for p in 18431 18432 18433; do bash stop_vllm.sh $p; done
G="$P/results/main3/comm7*/*.json,$P/results/main3/comm8/*.json,$P/results/locomo/comm7*/*.json,$P/results/locomo/comm8/*.json"
bash start_srv.sh 0 18441 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q14.log &
bash start_srv.sh 1 18442 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q14.log &
bash start_srv.sh 2 18443 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q14.log &
wait
$PY judge.py --model llama31 --port 18441 --glob "$G" > $P/logs/q14_j1.log 2>&1 &
$PY judge.py --model qwen25-7b --port 18442 --glob "$G" > $P/logs/q14_j2.log 2>&1 &
$PY judge.py --model qwen3-8b --port 18443 --glob "$G" > $P/logs/q14_j3.log 2>&1 &
wait
log judged
$PY comm_eval2.py --model llama31 --port 18441 --suffix _v9 --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,comm5,comm6,comm7,comm8,recompress,rebuild >> $P/logs/q14_comm.log 2>&1
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
CM_EMB_DEVICE=cpu $PY extras6.py > $P/results/extras9.txt 2>&1; cp $P/results/extras6.json $P/results/extras9.json
log "ALL DONE"; touch $P/logs/Q13.done
