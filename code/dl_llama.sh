#!/bin/bash
cd /data/yaokang/corrmem
HF_ENDPOINT=https://hf-mirror.com setsid nohup .venv/bin/huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --local-dir /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ > logs/dl_llama2.log 2>&1 < /dev/null &
echo dl
