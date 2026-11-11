#!/bin/bash
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2
for p in $(pgrep -f "post_q3.sh|queue3.sh|final.sh"); do kill $p; done
curl -s localhost:18431/v1/models | grep -q '"qwen3-8b"' || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74
CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu $PY run_corr.py --splits test200 --tag main --strategies ncf --clf_tag v2 --inst 12 --conc 64 > $P/logs/main_ncf_r3.log 2>&1 &
CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu $PY locomo.py run --methods ncf > $P/logs/locomo_ncf2.log 2>&1 &
wait
echo "ncf=$(ls $P/results/main/ncf | wc -l) locomo_ncf=$(ls $P/results/locomo/ncf | wc -l)" > $P/logs/final2.log
sed -i 's#^CM_CLE_DEVICE=cuda .*locomo.py run --methods ncf.*#true#' post_q3.sh
touch $P/logs/Q3.done
bash post_q3.sh
echo done >> $P/logs/final2.log
