#!/bin/bash
# Stop the old queue scripts only; their running A-Mem worker keeps going.
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null || true)
  case "$cmd" in
    "bash code/j_pub_amem.sh"*|"bash code/j_pub_resume.sh"*) echo "stop ${d##*/} $cmd"; kill "${d##*/}" 2>/dev/null || true ;;
  esac
done 2>/dev/null
