"""Rebuild full-set q30 tables without intersecting partial official baselines."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/data/yaokang/corrmem')
METHODS = 'append,slots,simupdate,coarse,ncf,spec,recompress,rebuild,comm10,comm20,comm20_nw,comm20_cite,comm20_nv,comm20_nr,comm20_overwrite,comm20_zs,rawrag,membank,mem0,oracle'


def main():
    if not (BASE / 'logs/Q30.done').exists():
        raise RuntimeError('q30 is not finished')
    env = dict(os.environ)
    env.pop('CM_RUN_ROOT', None)
    for judge in ('qwen25-7b', 'llama31', 'qwen3-8b'):
        for prefix, extra in [('t30', []), ('t30f', ['--valid_key', 'label_qwen25-7b'])]:
            dest = BASE / 'results' / f'{prefix}_main_{judge}.json'
            command = [sys.executable, 'analyze.py', '--tag', 'main3', '--methods', METHODS,
                       '--ref', 'comm20', '--key', 'label_' + judge, '--json', str(dest)] + extra
            subprocess.run(command, cwd=BASE / 'code', env=env, check=True)
    subprocess.run([sys.executable, 'analyze.py', '--tag', 'main3', '--methods', METHODS,
                    '--ref', 'comm20', '--key', 'label_qwen25-7b', '--all', '--json',
                    str(BASE / 'results/t30all_main_qwen25-7b.json')], cwd=BASE / 'code', env=env, check=True)
    # Package real seed-zero outputs alongside the later seeds for local aggregation.
    target = BASE / 'multiseed_v20/seed0/results'
    target.mkdir(parents=True, exist_ok=True)
    patterns = ('t30*.json', 'stat30.json', 'reg30.json', 'xtab.json', 'extras30.json',
                'suffbench_v20*.json', 'latency_v20.json', 'witness_stats.json',
                'comm2_main3_comm20*.json', 'comm2_main3_v20.json', 'cov_main3_comm20*.json',
                'coverage_main3_v20.json', 'tnet_eval_tnet20*_v20.json', 'judgecal_*.json',
                'label_cost_v20.json', 'train_meta_tnet20*.json')
    for pattern in patterns:
        for source in (BASE / 'results').glob(pattern):
            shutil.copy2(source, target / source.name)
    (target.parent / 'inputs.json').write_text(json.dumps({'training_seed': 0, 'source': 'q30',
        'main_table_protocol': 'Full fixed set; partial official baselines retained only in separate blocks'}, indent=2))


if __name__ == '__main__':
    main()
