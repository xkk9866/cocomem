#!/bin/bash
# Finish A-Mem (LongMemEval 201 questions, LoCoMo 10 conversations) on any card with room (GPUs 0-7).
# Servers that disappear are restarted elsewhere; interrupted runs resume (finished files are skipped,
# finished calls come from the response cache). Then judge with Qwen2.5-7B and export the table data.
set -Euo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
Q8=/home/yaokang/models/Qwen3-8B-AWQ
J7=/data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ
log() { echo "[$(date '+%H:%M:%S')] $*"; }
nfiles() { ls "$1" 2>/dev/null | grep -c 'json$' || true; }
up() { curl -sf localhost:$1/v1/models >/dev/null 2>&1; }
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v need="$1" '{f=$3-$2; if (f>=need) print f, $1, $3}' | sort -rn
}
ensure() {   # ensure PORT MODEL NAME : keep one server on PORT, on whichever card has room
  local pt=$1 m=$2 n=$3 f g t F
  exec 8>$P/logs/.srv_$pt.lock
  flock 8
  until up $pt; do
    while read -r f g t; do
      F=$(python3 -c "print(round(max(0.28, min(0.40, ($f-1800)/$t)), 3))")
      log "start $n port $pt gpu $g free $f frac $F"
      bash start_srv.sh $g $pt $m $n $F 8192 && break
      bash stop_vllm.sh $pt || true
    done < <(free_gpus 8800)
    up $pt || sleep 60
  done
  flock -u 8
}

extra() {   # one more replica on each card that is mostly free; these are not restarted if they go away
  local pt f g t F
  for pt in 18624 18625; do
    up $pt && continue
    read -r f g t < <(free_gpus 14000 | head -n 1) || return 0
    [ -n "${g:-}" ] || return 0
    F=$(python3 -c "print(round(min(0.82, ($f-2000)/$t), 3))")
    log "extra replica port $pt gpu $g free $f frac $F"
    bash start_srv.sh $g $pt $Q8 qwen3-8b $F 8192 || bash stop_vllm.sh $pt || true
  done
}
replicas() { local p out=""; for p in 18624 18625 18622 18623; do up $p && out="$out,$p"; done; echo "${out#,}"; }

ensure 18622 $Q8 qwen3-8b
ensure 18623 $Q8 qwen3-8b
extra
emb_gpu() {   # a card with at least 2.5GB free for the small embedder, or empty
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' '{f=$3-$2; if (f>=2500) print f, $1}' | sort -rn | awk 'NR<=2{print $2}' | tr '\n' ' '
}
ramcache() {   # fresh response and embedding caches on the RAM disk: the data disk is saturated by other jobs
  local D=/dev/shm/cm_cache_fresh_$USER
  mkdir -p $D
  echo $D
}
lme() {
  local round=0 C
  C=$(ramcache | tail -n 1)
  until [ "$(nfiles $P/results/main3/amem)" -ge 201 ]; do
    round=$((round+1))
    ensure 18622 $Q8 qwen3-8b; ensure 18623 $Q8 qwen3-8b
    R=$(replicas)
    read -r G1 G2 <<<"$(emb_gpu)"
    log "lme round $round at $(nfiles $P/results/main3/amem)/201 replicas $R emb gpus '$G1 $G2' cache $C"
    local i dev vis
    for i in 0 1 2 3 4 5 6 7 8 9; do
      dev=cpu; vis=""
      if [ "$round" -le 2 ] && [ -n "${G1:-}" ]; then
        dev=cuda; vis=$G1; [ -n "${G2:-}" ] && [ $((i % 2)) -eq 1 ] && vis=$G2
      fi
      CM_CACHE=$C CUDA_VISIBLE_DEVICES=$vis CM_EMB_DEVICE=$dev CM_P8=${R%%,*} CM_Q8_REPLICAS=$R \
        $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 2 --conc 8 --shard $i:10 \
        >> $P/logs/pub_am_fin_$i.log 2>&1 &
    done
    wait
    sleep 10
  done
  log "lme done"
}
loc() {
  until [ "$(nfiles $P/results/locomo/amem)" -ge 10 ]; do
    ensure 18623 $Q8 qwen3-8b; ensure 18622 $Q8 qwen3-8b
    R=$(replicas)
    log "locomo run at $(nfiles $P/results/locomo/amem)/10 replicas $R"
    CM_P8=${R%%,*} CM_Q8_REPLICAS=$R $PY locomo.py run --methods amem >> $P/logs/pub_locomo_amem.log 2>&1 || true
    sleep 10
  done
  log "locomo done"
}
lme & A=$!
loc & B=$!
wait $A $B
touch $P/logs/AMEM.done
for pt in 18622 18623 18624 18625; do bash stop_vllm.sh $pt || true; done

GLOB="$P/results/main3/amem/*.json,$P/results/locomo/amem/*.json,$P/results/main3/ga/*.json,$P/results/main3/readagent/*.json,$P/results/locomo/ga/*.json,$P/results/locomo/readagent/*.json"
until [ "$(grep -L 'label_qwen25-7b' $P/results/main3/amem/*.json $P/results/locomo/amem/*.json 2>/dev/null | wc -l)" -eq 0 ]; do
  ensure 18620 $J7 qwen25-7b
  $PY judge.py --model qwen25-7b --port 18620 --glob "$GLOB" >> $P/logs/pub_judge_final.log 2>&1 || true
  log "judge pass, unjudged $(grep -L 'label_qwen25-7b' $P/results/main3/amem/*.json $P/results/locomo/amem/*.json 2>/dev/null | wc -l)"
done
bash stop_vllm.sh 18620 || true
$PY export_pub.py > $P/logs/pub_export.log 2>&1
log "export: $(tail -n 1 $P/logs/pub_export.log)"
log PUB_DONE
touch $P/logs/PUB.done
