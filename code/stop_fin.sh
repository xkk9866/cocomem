#!/bin/bash
# Stop the finisher scripts and their A-Mem workers; servers stay up and are reused.
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cmd" in
    "bash code/j_pub_fin"*|*"run_sys.py --systems amem"*|*"locomo.py run --methods amem"*)
      echo "stop ${d##*/}"; kill "${d##*/}" 2>/dev/null || true ;;
  esac
done 2>/dev/null
