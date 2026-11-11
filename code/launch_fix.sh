#!/bin/bash
cd /data/yaokang/corrmem/code
for s in self_span ncf; do CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu CUDA_VISIBLE_DEVICES=2 setsid nohup ../.venv/bin/python run_corr.py --splits test200 --tag main --strategies $s --clf_tag v2 --inst 10 --conc 64 > /data/yaokang/corrmem/logs/main_${s}_r.log 2>&1 < /dev/null & done
echo ok
