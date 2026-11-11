#!/bin/bash
# Stop only the A-Mem generation processes, not the other baseline queue.
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *j_pub_amem.sh*|*"--systems amem"*|*"port 18622"*|*"port 18623"*)
      echo "stop ${d##*/}"
      kill "${d##*/}" 2>/dev/null || true
      ;;
  esac
done
sleep 6
