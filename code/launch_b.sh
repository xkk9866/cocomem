#!/bin/bash
cd /data/yaokang/corrmem/code
setsid nohup bash jobs_b.sh > /data/yaokang/corrmem/logs/jobs_b.out 2>&1 < /dev/null &
echo launched
