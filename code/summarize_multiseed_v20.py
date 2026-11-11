"""Summarize three training seeds; observations are seed-level metric values."""
import argparse
import json
import math
from pathlib import Path
import statistics

METHODS = tuple(json.loads(Path(__file__).with_name('multiseed_v20.json').read_text())['lme_methods'])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(root):
    values, counts = {}, {}
    rows = {r['method']: r for r in read(root / 't30_main_qwen25-7b.json')}
    for method in METHODS:
        value, n = rows[method]['corr']
        key = f'qa/{method}/corr'
        values[key], counts[key] = value, n
    if any(not math.isfinite(v) for v in values.values()):
        raise ValueError(f'Nonfinite metric in {root}; do not silently discard it')
    return values, counts


def stats(values):
    assert len(values) == 3
    return {'seeds': dict(zip(('0', '1', '2'), values)),
            'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}


def summarize(root, out):
    runs = [metrics(root / f'seed{s}/results') for s in range(3)]
    keys = set(runs[0][0])
    if any(set(values) != keys for values, _ in runs):
        raise ValueError('Metric sets differ across seeds')
    if any(counts != runs[0][1] for _, counts in runs):
        raise ValueError('Metric denominators differ across seeds')
    summary = {key: stats([run[0][key] for run in runs]) for key in sorted(keys)}
    paired = {}
    for key in sorted(keys):
        family, method, metric = key.split('/')
        if method == 'comm20':
            continue
        main = f'{family}/comm20/{metric}'
        if main in keys:
            paired[key] = stats([run[0][main] - run[0][key] for run in runs])
    report = {
        'protocol': 'Three independent training seeds (0,1,2), conditional on fixed data, witnesses, evaluation questions, oracle-valid mask for LongMemEval-Corr QA. SD uses ddof=1. Questions are not pooled across seeds as independent observations.',
        'paired_difference': 'Within each seed: main method minus ablation, in the reported metric units. Positive is favorable only for higher-is-better metrics.',
        'metrics': summary, 'paired_ablations': paired, 'denominators': runs[0][1],
        'frozen_baselines': [row for row in read(root / 'seed0/results/t30_main_qwen25-7b.json') if not row['method'].startswith('comm20')],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(summary)} metrics and {len(paired)} paired ablation differences: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summarize(args.root, args.out)
