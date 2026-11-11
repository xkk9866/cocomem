#!/bin/bash
# repair after the concurrent-judge collision: find broken result files, regenerate them, re-judge, re-analyse.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q16.log; }
log start
find $P/results/main3 $P/results/locomo -name '*.tmp' -delete; find $P/results -name '*.lock' -delete
$PY - <<'PYEOF' > $P/logs/q16_broken.txt
import glob, json
for f in sorted(glob.glob("/data/yaokang/corrmem/results/main3/comm[78]*/*.json") + glob.glob("/data/yaokang/corrmem/results/locomo/comm[78]*/*.json")):
    try: json.load(open(f))
    except Exception: print(f)
PYEOF
log "broken $(wc -l < $P/logs/q16_broken.txt)"
for f in $(cat $P/logs/q16_broken.txt); do rm -f $f; done
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
for p in 18441 18442 18443; do bash stop_vllm.sh $p; done
if true; then
  bash start_srv.sh 2 18431 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.55 >> $P/logs/q16.log
  bash start_tnet.sh $P/models/tnet7/merged 0.18 tnet7 18432 >> $P/logs/q16.log
  bash start_tnet.sh $P/models/tnet3/merged 0.18 tnet3 18433 2 >> $P/logs/q16.log
  up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
  up 18431 qwen3-8b && up 18432 tnet7 && up 18433 tnet3 || { log SERVERS_DOWN; touch $P/logs/Q16.done; exit 1; }
  for s in comm7 comm7_overwrite comm8; do $PY run_corr.py --splits test200 --tag main3 --strategies $s --inst 10 --conc 96 >> $P/logs/m3_${s}_fix.log 2>&1 & done
  CM_PT=18432 $PY locomo.py run --methods comm7,comm8,comm7_overwrite >> $P/logs/locomo_fix.log 2>&1
  wait
  for p in 18431 18432 18433; do bash stop_vllm.sh $p; done
fi
G="$P/results/main3/comm7*/*.json,$P/results/main3/comm8/*.json,$P/results/locomo/comm7*/*.json,$P/results/locomo/comm8/*.json"
bash start_srv.sh 0 18441 /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q16.log &
bash start_srv.sh 1 18442 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q16.log &
bash start_srv.sh 2 18443 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80 >> $P/logs/q16.log &
wait
$PY judge.py --model llama31 --port 18441 --glob "$G" > $P/logs/q16_j1.log 2>&1 &
$PY judge.py --model qwen25-7b --port 18442 --glob "$G" > $P/logs/q16_j2.log 2>&1 &
$PY judge.py --model qwen3-8b --port 18443 --glob "$G" > $P/logs/q16_j3.log 2>&1 &
wait
log judged
$PY comm_eval2.py --model llama31 --port 18441 --suffix _v9 --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,comm5,comm6,comm7,comm8,recompress,rebuild >> $P/logs/q16_comm.log 2>&1 &
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm2,comm3,comm3_nofallback,comm4,comm5,comm6,comm7,comm7_overwrite,comm8,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm7 --key label_$k --json $P/results/t13_main_$k.json > $P/results/t13_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm7 --key label_$k --valid_key label_qwen25-7b --json $P/results/t13f_main_$k.json > $P/results/t13f_main_$k.txt 2>&1
done
L=comm7,comm7_overwrite,comm8,comm5,comm6,comm4,comm3,comm2,comm,comm_zs,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t13_locomo_$k.json > $P/results/t13_locomo_$k.txt 2>&1
done
CM_EMB_DEVICE=cpu $PY extras6.py > $P/results/extras9.txt 2>&1; cp $P/results/extras6.json $P/results/extras9.json
wait
[ -s $P/results/comm2_main3_v9.json ] && log "ALL DONE" || log "COMM_EVAL_MISSING"
touch $P/logs/Q16.done
