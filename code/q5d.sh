#!/bin/bash
# v4 stage D: resume q5c from the strategy runs (servers already up on GPU 2; embedding clients on GPU 5).
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q5.log; }
log "D start"
curl -s localhost:18432/v1/models | grep -q tnet && curl -s localhost:18431/v1/models | grep -q qwen3-8b || { log "SERVERS_DOWN abort"; exit 1; }
export CUDA_VISIBLE_DEVICES=5 CM_EMB_DEVICE=cuda
[ -e $P/results/jacobi2.json ] || $PY jacobi2.py > $P/logs/q5_jacobi2.log 2>&1 &
$PY run_corr.py --splits test200 --tag main3 --strategies comm,comm_overwrite,comm_zs --inst 12 --conc 128 >> $P/logs/m3_comm.log 2>&1 &
$PY locomo.py run --methods comm,comm_zs,comm_overwrite >> $P/logs/locomo_v4.log 2>&1 &
CM_EMB_DEVICE=cpu $PY run_sys.py --systems oracle --splits test200 --tag main3 --inst 8 >> $P/logs/m3_oracle.log 2>&1 &
wait
for s in comm comm_overwrite comm_zs oracle; do log "main3 $s $(ls $P/results/main3/$s | wc -l)"; done
for s in comm comm_zs comm_overwrite; do log "locomo $s $(ls $P/results/locomo/$s 2>/dev/null | wc -l)"; done
log "stage1 done"
export CUDA_VISIBLE_DEVICES=2
bash stop_vllm.sh 18432; bash stop_vllm.sh 18431
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator.py score >> $P/logs/q5_tnet.log 2>&1
log "score done"
bash start_gpu2.sh /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ llama31 0.80 >> $P/logs/q5.log
$PY comm_eval2.py --model llama31 > $P/logs/q5_comm2.log 2>&1
$PY judge.py --model llama31 --glob "$P/results/main3/comm*/*.json,$P/results/main3/oracle/*.json,$P/results/locomo/comm*/*.json" > $P/logs/q5_judge_llama.log 2>&1
log "llama done"
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.80 >> $P/logs/q5.log
$PY judge.py --model qwen25-7b --glob "$P/results/main3/comm*/*.json,$P/results/main3/oracle/*.json,$P/results/locomo/comm*/*.json" > $P/logs/q5_judge_q25.log 2>&1
log "qwen25 done"
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/q5.log
$PY judge.py --model qwen3-8b --glob "$P/results/main3/comm*/*.json,$P/results/main3/oracle/*.json,$P/results/locomo/comm*/*.json" > $P/logs/q5_judge_q3.log 2>&1
M=append,slots,simupdate,self,self_span,coarse,ncf,ccf,recompress,spec,rebuild,comm_zs,comm,comm_overwrite,rawrag,membank,mem0,oracle
L=comm,comm_zs,comm_overwrite,ncf,ccf,spec,append,slots,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm --key label_$k --json $P/results/t5_main_$k.json > $P/results/t5_main_$k.txt 2>&1
  $PY analyze.py --tag locomo --methods $L --key label_$k --json $P/results/t5_locomo_$k.json > $P/results/t5_locomo_$k.txt 2>&1
done
CM_EMB_DEVICE=cpu $PY extras.py --tag main3 > $P/results/extras5.txt 2>&1
log "ALL DONE"; touch $P/logs/Q5B.done
