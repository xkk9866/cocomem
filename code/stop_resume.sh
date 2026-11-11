#!/bin/bash
# Stop only the waiting published-baseline resume loop.
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *j_pub_resume.sh*) kill "${d##*/}" 2>/dev/null && echo stopped "${d##*/}" ;;
  esac
done
