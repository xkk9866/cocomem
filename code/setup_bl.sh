#!/bin/bash
# Environments for the OFFICIAL baseline code (CPU only; the LLM is the shared vLLM server).
#   venv_m0 : mem0ai==1.0.2 (paper-era ADD/UPDATE/DELETE pipeline) + A-Mem (agiresearch/A-mem, chromadb)
#   venv_m2 : mem0ai latest (v2 additive pipeline)
#   venv_lm : langmem==0.0.29
P=/data/yaokang/corrmem; export PATH=$HOME/.local/bin:$PATH
mkdir -p $P/bl; cd $P/bl
mk() { [ -x $P/bl/$1/bin/python ] || uv venv --python 3.12 $P/bl/$1; }
mk venv_m0; mk venv_m2; mk venv_lm
U="uv pip install --python"
$U $P/bl/venv_m0/bin/python "mem0ai==1.0.2" "litellm==1.75.9" chromadb sentence-transformers rank_bm25 nltk openai qdrant-client scikit-learn numpy > $P/logs/setup_m0.log 2>&1 && echo m0 ok >> $P/logs/setup_bl.log
$U $P/bl/venv_m2/bin/python mem0ai sentence-transformers openai qdrant-client fastembed spacy > $P/logs/setup_m2.log 2>&1 && echo m2 ok >> $P/logs/setup_bl.log
$U $P/bl/venv_lm/bin/python "langmem==0.0.29" "litellm==1.75.9" langchain-huggingface sentence-transformers openai > $P/logs/setup_lm.log 2>&1 && echo lm ok >> $P/logs/setup_bl.log
touch $P/logs/SETUP_BL.done
