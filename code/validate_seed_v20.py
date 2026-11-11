"""Reject incomplete seed runs before aggregation; never filter based on scores."""
import argparse
import json
from pathlib import Path

from prepare_multiseed_v20 import BASE, VARIANTS, digest


def read(path):
    return json.loads(path.read_text())


def validate(root, seed):
    manifest = read(root / 'inputs.json')
    assert manifest['training_seed'] == seed
    for variant in VARIANTS:
        meta = read(root / 'models' / variant / 'train_meta.json')
        assert meta['seed'] == seed, (variant, 'wrong training seed')
        assert (root / 'models' / variant / 'merged/config.json').is_file()
        for split in ('train', 'test'):
            assert digest(root / 'models' / variant / f'{split}.jsonl') == manifest['training_sha256'][f'{variant}/{split}']
    config = read(BASE / 'code/multiseed_v20.json')
    for tag, key in [('main3', 'lme_methods'), ('locomo', 'locomo_methods'), ('locomo_reg', 'recurrence_methods')]:
        for method in config[key]:
            source = BASE / 'results' / tag / method
            dest = root / 'results' / tag / method
            assert not dest.is_symlink(), (tag, method, 'student results cannot be a baseline link')
            expected = {f.name for f in source.glob('*.json')}
            actual = {f.name for f in dest.glob('*.json')}
            assert expected and actual == expected, (tag, method, 'question set differs', len(expected), len(actual))
            for name in sorted(expected):
                zero, current = read(source / name), read(dest / name)
                keys = lambda d: {(a['scenario'], a['qkey']) for a in d['answers']}
                assert keys(zero) == keys(current), (tag, method, name, 'answer set differs')
                inputs = lambda d: {(a['scenario'], a['qkey']): (a['question'], a['gold'], a['jtype']) for a in d['answers']}
                assert inputs(zero) == inputs(current), (tag, method, name, 'questions or targets differ')
                assert all('label_qwen25-7b' in a for a in current['answers']), (tag, method, name, 'missing judge label')
    table = read(root / 'results/t30_main_qwen25-7b.json')
    zero = read(BASE / 'multiseed_v20/seed0/results/t30_main_qwen25-7b.json')
    baseline = {row['method']: row for row in zero}
    rows = {row['method']: row for row in table}
    for method in config['lme_methods']:
        assert rows[method]['n'] == baseline[method]['n']
        assert rows[method]['corr'][1] == baseline[method]['corr'][1]
    print(f'seed={seed}: complete; fixed question sets and training inputs verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    validate(args.root, args.seed)
