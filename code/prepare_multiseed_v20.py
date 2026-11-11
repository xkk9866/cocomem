"""Create isolated seed runs after q30; freeze inputs and record their hashes."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

BASE = Path('/data/yaokang/corrmem')
VARIANTS = ('tnet20', 'tnet20nw', 'tnet20nv', 'tnet20nr')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def link(source, target):
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            raise RuntimeError(f'Existing path points elsewhere: {target}')
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def prepare(seed):
    if not (BASE / 'logs/Q30.done').is_file():
        raise RuntimeError('q30 must complete before preparing new seed runs')
    if seed not in (1, 2):
        raise ValueError('Only new seeds 1 and 2 are allowed')
    root = BASE / 'multiseed_v20' / f'seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    for name in ('results', 'models', 'cache', 'logs', 'steps'):
        (root / name).mkdir(exist_ok=True)
    # Set WAL once before parallel evaluators open a new embedding cache.
    db = sqlite3.connect(root / 'cache' / 'emb.sqlite', timeout=120)
    try:
        if db.execute('PRAGMA journal_mode=WAL').fetchone()[0] != 'wal':
            raise RuntimeError('Embedding cache could not enable WAL')
    finally:
        db.close()
    for name in ('data', 'states'):
        link(BASE / name, root / name)
    hashes = {}
    for variant in VARIANTS:
        dest = root / 'models' / variant
        dest.mkdir(exist_ok=True)
        for split in ('train', 'test'):
            source = BASE / 'models' / variant / f'{split}.jsonl'
            hashes[f'{variant}/{split}'] = digest(source)
            link(source, dest / source.name)
    # Frozen comparators and the oracle mask are shared, never relabeled here.
    for tag in ('main3', 'locomo', 'locomo_reg'):
        dest = root / 'results' / tag
        dest.mkdir(exist_ok=True)
        for source in sorted((BASE / 'results' / tag).iterdir()):
            if source.is_dir() and not source.name.startswith('comm20'):
                link(source, dest / source.name)
    manifest = {'training_seed': seed, 'benchmark_seed': 0,
                'training_sha256': hashes, 'source': str(BASE),
                'student_suffix': f'_s{seed}', 'root': str(root)}
    path = root / 'inputs.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError(f'Inputs changed since the run was prepared: {path}')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int)
    prepare(parser.parse_args().seed)
