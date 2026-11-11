#!/bin/bash
# Start j_pub_fin3.sh once.
cd /data/yaokang/corrmem
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cmd" in "bash code/j_pub_fin3.sh"*) echo "already running ${d##*/}"; exit 0 ;; esac
done 2>/dev/null
setsid nohup bash code/j_pub_fin3.sh > logs/j_pub_fin3.log 2>&1 < /dev/null &
echo "started $!"
