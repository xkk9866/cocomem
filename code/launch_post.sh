#!/bin/bash
cd /data/yaokang/corrmem/code
setsid nohup bash post_q3.sh > /data/yaokang/corrmem/logs/post_q3.out 2>&1 < /dev/null &
echo launched
