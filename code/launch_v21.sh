#!/bin/bash
# Start (or restart) the v21 scheduler.  Idempotent: finished jobs (logs/J_*.done) are skipped, running jobs of a
# previous scheduler instance are adopted.  GPUs: SCHED_GPUS (default 0-7); only the memory free at launch is used.
#   bash code/launch_v21.sh            # start
#   touch logs/SCHED21.pause           # no new launches;  rm to resume
#   touch logs/SCHED21.stop            # stop everything (the scheduler kills its jobs and exits)
#   touch logs/SCHED21.reload          # the scheduler exits without killing jobs; run launch_v21.sh again to reload code
P=/data/yaokang/corrmem; cd $P/code
rm -f $P/logs/SCHED21.stop $P/logs/SCHED21.pause $P/logs/SCHED21.done $P/logs/SCHED21.reload
if pgrep -u "$USER" -f '[s]ched21.py' > /dev/null; then
  echo "a scheduler is already running: $(pgrep -u "$USER" -f '[s]ched21.py' | tr '\n' ' ')"; exit 0
fi
SCHED_GPUS=${SCHED_GPUS:-0,1,2,3,4,5,6,7} setsid nohup ../.venv/bin/python sched21.py > $P/logs/sched21_stdout.log 2>&1 < /dev/null &
sleep 3
echo "started sched21 pid=$! gpus=${SCHED_GPUS:-0,1,2,3,4,5,6,7}"; tail -n 2 $P/logs/sched21.log
