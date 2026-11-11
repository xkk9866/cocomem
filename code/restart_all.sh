#!/bin/bash
# Clean restart of the whole pipeline on GPU 2: stop post/queue scripts and clients, start the agent LLM,
# then launch queue3 (experiments) and post_q3 (A-Mem + judging + tables, waits for logs/Q3.done).
P=/data/yaokang/corrmem; cd $P/code
for p in $(pgrep -f "post_q3.sh|queue3.sh"); do kill $p; done
for p in $(pgrep -f "run_corr.py|run_sys.py|locomo.py run|judge.py|lineage_all.py"); do kill $p; done
rm -f $P/logs/Q3.done $P/logs/POST.done
sleep 3
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74
setsid nohup bash queue3.sh > $P/logs/queue3.out 2>&1 < /dev/null &
CM_EMB_DEVICE=cpu setsid nohup bash post_q3.sh > $P/logs/post_q3.out 2>&1 < /dev/null &
echo restarted
