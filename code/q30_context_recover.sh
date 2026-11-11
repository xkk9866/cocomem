#!/bin/bash
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
exec 8>"$P/logs/q30.context_recovery.lock"
flock -n 8 || exit 1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$P/logs/q30_context_recovery.log"; }
log 'waiting for existing programmatic job 3029671; queue 1696327 suspended at wait'
while [ -r /proc/3029671/cmdline ] && grep -aq 'suffbench4.py' /proc/3029671/cmdline; do sleep 20; done
if ../.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/data/yaokang/corrmem/results/suffbench_v20.json')
d=json.loads(p.read_text())
assert d['chain']['length']==100
assert all(set(d['chain'][m])=={'1','10','50','100'} for m in ('comm20','comm20_nw','ledger_all','copy'))
PY
then touch "$P/logs/q30_suff_a.done"; log 'completed programmatic output retained'; fi
if [ -r /proc/1696327/cmdline ] && grep -aq 'bash.q30.sh' /proc/1696327/cmdline; then
  kill -TERM 1696327
  kill -CONT 1696327
fi
for i in $(seq 1 30); do
  if [ ! -r /proc/1696327/cmdline ] || ! grep -aq 'bash.q30.sh' /proc/1696327/cmdline; then break; fi
  sleep 2
done
bash stop_vllm.sh 18432
bash stop_vllm.sh 18433
cp q30_ctx_resume.sh q30.sh.next
mv q30.sh.next q30.sh
log 'resuming cached q30 with 16384-token student contexts and explicit child status checks'
flock -u 8
exec 8>&-
export Q30_RESUME_EVAL=1
exec bash q30.sh
