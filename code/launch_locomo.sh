#!/bin/bash
cd /data/yaokang/corrmem/code
CM_EMB_DEVICE=cpu CUDA_VISIBLE_DEVICES=2 setsid nohup ../.venv/bin/python locomo.py run --methods spec,append,simupdate,coarse,rawrag,membank,self,recompress,rebuild > /data/yaokang/corrmem/logs/locomo_early.log 2>&1 < /dev/null &
echo launched
