#!/bin/bash
P=/data/yaokang/corrmem; cd $P/code
for p in $(pgrep -f "post_q3.sh|judge.py"); do kill $p; done
rm -f $P/logs/Q3.done
CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python run_corr.py --splits test200 --tag main --strategies ncf --clf_tag v2 --inst 12 --conc 64 > $P/logs/main_ncf_r2.log 2>&1
touch $P/logs/Q3.done
bash post_q3.sh
