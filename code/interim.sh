#!/bin/bash
cd /data/yaokang/corrmem/code
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
../.venv/bin/python judge.py --model qwen3-8b --glob "/data/yaokang/corrmem/results/main/*/*.json,/data/yaokang/corrmem/results/locomo/*/*.json" > ../logs/judge_interim.log 2>&1
M=append,simupdate,self,self_span,coarse,recompress,spec_ncf,spec_overwrite,rebuild,spec,rawrag,membank,mem0
../.venv/bin/python analyze.py --tag main --methods $M --ref spec --key label_qwen3-8b > ../results/interim_main.txt 2>&1
../.venv/bin/python analyze.py --tag locomo --methods spec,append,simupdate,self,coarse,recompress,rebuild,rawrag,membank,amem --key label_qwen3-8b > ../results/interim_locomo.txt 2>&1
touch ../logs/interim.done
