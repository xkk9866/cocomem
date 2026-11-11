#!/bin/bash
# stop queue2 and all LLM client jobs (keeps the vLLM server)
for p in $(pgrep -f "queue[23].sh"); do kill $p; done
for p in $(pgrep -f "run_corr.py|run_sys.py|locomo.py run|lineage_all.py|judge.py"); do kill $p; done
sleep 3; echo stopped
