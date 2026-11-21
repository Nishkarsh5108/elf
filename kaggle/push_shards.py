"""Push the engine as a private Kaggle dataset and launch N private CPU kernels, one shard each.

usage:
  python kaggle/push_shards.py dataset            # (re)upload private dataset <user>/elf-engine (cpp + puzzle files)
  python kaggle/push_shards.py launch NAME NSHARDS "ENGINE ARGS" [puzzles=p_all.txt]
  python kaggle/push_shards.py status NAME NSHARDS
  python kaggle/push_shards.py fetch NAME NSHARDS OUTDIR
Kernels and dataset are private. Kernel slugs: <user>/elf-NAME-<k>.
"""
import json, os, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(ROOT, '.venv', 'Scripts', 'kaggle.exe')
if not os.path.exists(K): K = 'kaggle'
USER = json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username']


def sh(args):
    print('>', ' '.join(args[:6]), flush=True)
    return subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')


def dataset():
    d = os.path.join(ROOT, 'kaggle', 'engine_ds')
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    shutil.copy(os.path.join(ROOT, 'cpp', 'elf.cpp'), d)
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from santa import load_puzzles
    from io_txt import write_puzzles_txt
    write_puzzles_txt(os.path.join(d, 'p_all.txt'), load_puzzles())
    meta = {'title': 'elf-engine', 'id': f'{USER}/elf-engine', 'licenses': [{'name': 'CC0-1.0'}]}
    json.dump(meta, open(os.path.join(d, 'dataset-metadata.json'), 'w'))
    r = sh([K, 'datasets', 'status', f'{USER}/elf-engine'])
    if 'ready' in (r.stdout + r.stderr).lower():
        r = sh([K, 'datasets', 'version', '-p', d, '-m', 'update', '-r', 'zip'])
    else:
        r = sh([K, 'datasets', 'create', '-p', d, '-r', 'zip'])  # private by default
    print(r.stdout[-800:], r.stderr[-800:])


def launch(name, n, args, puzzles='p_all.txt', only=None):
    for k in (only if only is not None else range(n)):
        d = os.path.join(ROOT, 'kaggle', 'kernels', f'{name}-{k}')
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        cfg = json.dumps({'shard': k, 'nshards': n, 'args': args, 'puzzles': puzzles})
        code = open(os.path.join(ROOT, 'kaggle', 'run_kernel.py')).read().replace('__CONFIG__', cfg)
        open(os.path.join(d, 'run.py'), 'w').write(code)
        meta = {'id': f'{USER}/elf-{name}-{k}', 'title': f'elf-{name}-{k}', 'code_file': 'run.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
                'enable_internet': False, 'dataset_sources': [f'{USER}/elf-engine'],
                'competition_sources': ['santa-2026'], 'kernel_sources': []}
        json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'))
        r = sh([K, 'kernels', 'push', '-p', d])
        print(r.stdout[-400:], r.stderr[-400:])


def status(name, n, only=None):
    for k in (only if only is not None else range(n)):
        r = sh([K, 'kernels', 'status', f'{USER}/elf-{name}-{k}'])
        print(k, (r.stdout + r.stderr).strip()[-200:])


def fetch(name, n, out, only=None):
    os.makedirs(out, exist_ok=True)
    for k in (only if only is not None else range(n)):
        d = os.path.join(out, f'{name}-{k}')
        os.makedirs(d, exist_ok=True)
        r = sh([K, 'kernels', 'output', f'{USER}/elf-{name}-{k}', '-p', d])
        print(k, r.stdout[-200:], r.stderr[-200:])


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'dataset': dataset()
    elif c == 'launch':
        only = [int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else None
        launch(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else 'p_all.txt', only)
    elif c == 'status': status(sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else None)
    elif c == 'fetch': fetch(sys.argv[2], int(sys.argv[3]), sys.argv[4], [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else None)
