#!/bin/bash
P=/data/yaokang/corrmem
for p in $(pgrep -f "queue4.sh|comm_eval.py|judge.py"); do kill $p; done
rm -f $P/cache/llama31.sqlite* $P/results/comm_main3*.json
cd $P/code; setsid nohup bash stage23.sh > $P/logs/stage23.out 2>&1 < /dev/null &
echo relaunched
