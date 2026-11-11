#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
/data/yaokang/corrmem/.venv/bin/huggingface-cli download Qwen/Qwen3-1.7B --local-dir /data/yaokang/models/Qwen3-1.7B > /data/yaokang/corrmem/logs/dl_q17.log 2>&1
echo done >> /data/yaokang/corrmem/logs/dl_q17.log
