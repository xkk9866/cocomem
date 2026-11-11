#!/bin/bash
cd /data/yaokang/corrmem/code
rm -f ../logs/Q4.done
setsid nohup bash queue4.sh > ../logs/queue4.out 2>&1 < /dev/null &
echo launched
