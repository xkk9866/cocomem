#!/bin/bash
cd /data/yaokang/corrmem/code
setsid nohup bash queue3.sh > /data/yaokang/corrmem/logs/queue3.out 2>&1 < /dev/null &
echo launched
