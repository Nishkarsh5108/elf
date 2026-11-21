# Kaggle kernel entry point (pushed as a private script kernel).
# Compiles the engine from the attached private dataset and runs one shard of puzzles.
# Shard/config are baked in by kaggle/push_shards.py via the CONFIG line below.
import os, subprocess, glob, time, json
CONFIG = json.loads('__CONFIG__')
src = glob.glob('/kaggle/input/**/elf.cpp', recursive=True)[0]
puz = glob.glob('/kaggle/input/**/' + CONFIG['puzzles'], recursive=True)[0]
os.makedirs('/kaggle/working/out', exist_ok=True)
t0 = time.time()
subprocess.run(['g++', '-O3', '-march=native', '-std=c++20', '-pthread', src, '-o', '/kaggle/working/elf'], check=True)
lines = open(puz).read().splitlines()
k, n = CONFIG['shard'], CONFIG['nshards']
mine = [l for i, l in enumerate(lines) if i % n == k]
open('/kaggle/working/shard.txt', 'w').write('\n'.join(mine) + '\n')
nproc = os.cpu_count()
cmd = ['/kaggle/working/elf', '/kaggle/working/shard.txt', '-t', str(nproc)] + CONFIG['args'].split()
print('running', cmd, len(mine), 'puzzles', flush=True)
with open('/kaggle/working/out/result.txt', 'w') as fo, open('/kaggle/working/out/log.txt', 'w') as fe:
    subprocess.run(cmd, stdout=fo, stderr=fe)
print('done in %.0fs' % (time.time() - t0), flush=True)
print(open('/kaggle/working/out/log.txt').read()[-2000:])
