#!/bin/bash
# Stop this project's idle inference servers left from the finished run.
# Match the server command line, never this shell.
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *vllm.entrypoints*)
      case "$cmd" in
        *tnet21*|*18621*) echo "stop ${d##*/}"; kill "${d##*/}" 2>/dev/null || true ;;
      esac
      ;;
  esac
done
sleep 8
