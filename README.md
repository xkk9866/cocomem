# CoCoMem

Research code for correction-aware, self-maintainable long-term memory for LLM agents. The implementation builds compressed memory units with derivation records, applies corrections, and evaluates memory maintenance and question answering on LongMemEval-S and LoCoMo.

## Repository contents

- `code/cocomem.py`: memory construction and correction strategies.
- `code/deriv.py`, `code/ncf.py`, `code/commutator21.py`: derivation records, learned dependency model, and correction operator.
- `code/run_build.py`, `code/run_corr.py`, `code/run_sys.py`: experiment entry points.
- `code/sched22.py` and `code/*.sh`: cluster scheduling and experiment orchestration.
- `code/tools/`: synchronization and result collection helpers.

This repository contains source code and shell scripts only. Datasets, model weights, experiment outputs, and manuscript files are not included.

## Environment

The scripts target Linux with Python 3, CUDA GPUs, and locally served OpenAI-compatible model endpoints. Python dependencies used across the code include `numpy`, `torch`, `transformers`, `peft`, `openai`, `httpx`, `scikit-learn`, and `tqdm`. Install the dependencies needed by the entry point you run in a virtual environment.

Several scripts contain paths from the original compute environment, such as `/data/yaokang/corrmem` and `/home/yaokang/models`. Update these paths, model locations, ports, and GPU assignments for your machine. Some entry points also support `CM_RUN_ROOT`.

## Data and execution

Obtain LongMemEval-S and LoCoMo from their respective official sources and place the prepared data where `code/data.py` expects it, or update its path constants. Start the required model servers before running experiments. From the project root, inspect an entry point's options with, for example:

```bash
python code/run_corr.py --help
```

The shell scripts under `code/` encode specific experiment configurations and server assumptions. Review their paths and commands before using them on another cluster.

## Scope

This is a research snapshot rather than a packaged library. Reproducing reported experiments requires the external datasets, model checkpoints, and compute setup described above.
