#!/bin/bash
cd /data/yaokang/corrmem/code
setsid nohup bash queue2.sh > /data/yaokang/corrmem/logs/queue2.out 2>&1 < /dev/null &
echo launched
