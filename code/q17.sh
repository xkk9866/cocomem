#!/bin/bash
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
M=append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild,comm_zs,comm,comm5,comm5_overwrite,comm6,comm7,comm7_overwrite,comm8,rawrag,membank,mem0,oracle
for k in qwen25-7b llama31 qwen3-8b; do
  $PY analyze.py --tag main3 --methods $M --ref comm5 --key label_$k --json $P/results/t14_main_$k.json > $P/results/t14_main_$k.txt 2>&1
  $PY analyze.py --tag main3 --methods $M --ref comm5 --key label_$k --valid_key label_qwen25-7b --json $P/results/t14f_main_$k.json > $P/results/t14f_main_$k.txt 2>&1
done
$PY stat17.py > $P/results/stat17.txt 2>&1
touch $P/logs/Q17.done
