#!/bin/bash
# Start (or restart) the v22 scheduler (sched22.py on the sched21 core).  Idempotent: finished jobs (logs/J_v22_*.done)
# are skipped, running jobs of a previous instance are adopted.  GPUs: SCHED_GPUS (default 0-4; GPUs 5, 6, 7 are
# off limits for this project since 11-10 12:22).
#   bash code/launch_v22.sh            # start
#   touch logs/SCHED22.pause           # no new launches;  rm to resume
#   touch logs/SCHED22.stop            # stop everything (the scheduler kills its jobs and exits)
#   touch logs/SCHED22.reload          # exit without killing jobs; run launch_v22.sh again to reload code
P=/data/yaokang/corrmem; cd $P/code
SCHED_GPUS=${SCHED_GPUS:-0,1,2,3,4}
case ",$SCHED_GPUS," in *,5,*|*,6,*|*,7,*) echo "refusing SCHED_GPUS=$SCHED_GPUS: GPUs 5, 6, 7 are off limits"; exit 1 ;; esac
rm -f $P/logs/SCHED22.stop $P/logs/SCHED22.pause $P/logs/SCHED22.done $P/logs/SCHED22.reload
if pgrep -u "$USER" -f '[s]ched22.py' > /dev/null; then
  echo "a scheduler is already running: $(pgrep -u "$USER" -f '[s]ched22.py' | tr '\n' ' ')"; exit 0
fi
SCHED_GPUS=$SCHED_GPUS setsid nohup ../.venv/bin/python sched22.py > $P/logs/sched22_stdout.log 2>&1 < /dev/null &
sleep 3
echo "started sched22 pid=$! gpus=$SCHED_GPUS"; tail -n 2 $P/logs/sched22.log
