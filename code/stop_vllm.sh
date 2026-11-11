#!/bin/bash
# stop the vLLM server listening on port $1 (by port, never by a pattern that could match this shell)
L=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | cut -d= -f2)
[ -n "$L" ] && kill $L 2>/dev/null
for i in $(seq 1 30); do sleep 2; ss -ltn | grep -q ":$1 " || break; done
sleep 8
