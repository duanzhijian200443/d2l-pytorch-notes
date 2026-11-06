# Name That LLM - full solution: fold-averaged two-stage LoRA24 (Qwen3-8B-Base, Qwen3-14B-Base, Qwen2.5-7B),
# Nishant two-stage checkpoints, 4-view n-gram SVM on train+pool, shared-prompt Gaussian, adaptive centred twins,
# 10% public ATLAS (source-conditioned likelihood, arXiv 2610.19583) before Sinkhorn balancing.
import glob, os, subprocess, json, sys
whl = glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True)
if whl: print(subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
def mdir(*parts):
    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts)]
    assert c, parts; return c[0]
def adir(name):
    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True); assert h, name; return os.path.dirname(h[0])
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
import pandas as _pd
FAST_COMMIT = len(_pd.read_csv(DATA + '/test.csv')) < 50   # fork: the 3-row commit run skips the heavy cells; the hidden rerun runs everything
print('FAST_COMMIT', FAST_COMMIT)
HEADS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/heads/q3_8bx.npz', recursive=True))))
EXTRA = next(iter(glob.glob('/kaggle/input/**/nlm_matched_clean.parquet', recursive=True)))
POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))
B8, B14, B25 = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/'), mdir('/qwen2.5/', '/7b/')
I8, I14 = mdir('/qwen-3/', '/8b/'), mdir('/qwen-3/', '/14b/')
A8 = [adir(f'ts_q3b8_f{k}') for k in range(5)] + [adir('ts_q3b8')]
A14 = [adir(f'ts_q3b14_f{k}') for k in (0, 2, 4)] + [adir('ts_q3b14')]
A25 = [adir('ts_q25_7b')]
print(B8, B14, B25, I8, I14, A8, A14, A25, HEADS, EXTRA, POOL, sep='\n')
def nmeta(name):
    h = glob.glob(f'/kaggle/input/**/{name}/meta.json', recursive=True); assert h, name
    d = os.path.dirname(h[0]); c = json.load(open(h[0]))['cfg']; return d, c
N8, N8C = nmeta('lora8b_v1'); N14, N14C = nmeta('lora14b_v1')
print('nishant', N8, N8C['keep'], N8C['max_len'], N14, N14C['keep'], N14C['max_len'])
ATLAS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True))))
print('atlas', ATLAS)
BL = mdir('/llama31-8b-base-verified/')
_SHARDS = {'model-00001-of-00004.safetensors': 4976698672, 'model-00002-of-00004.safetensors': 4999802720, 'model-00003-of-00004.safetensors': 4915916176, 'model-00004-of-00004.safetensors': 1168138808}
assert all(
    os.path.getsize(os.path.join(BL, f)) == n
    for f, n in _SHARDS.items()
), BL   # fork: Llama-3.1-8B from avazaku's public cache (NousResearch copy)
AL = [adir(f'ts_llama8_f{k}') for k in (0, 2, 4)] + [adir('ts_llama8')]
import shutil, time as _tc; _c = _tc.time()
shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True); BL = '/tmp/llama-3.1/8b'   # fork: see the intro
print('llama copied to local disk in', round(_tc.time() - _c), 's')
print('llama', BL, AL)
A8S7 = [adir(f'ts8_s7_f{k}') for k in (0, 2, 4)] + [adir('ts8_s7')]
print('seed7', A8S7)
B25_14 = mdir('/qwen2.5/', '/14b/')
N14B, N14BC = nmeta('lora_qwen3_14b'); Q25_14, Q25_14C = nmeta('lora_qwen25_14b')
DEB = os.path.dirname(glob.glob('/kaggle/input/**/deberta_v3_large/meta.json', recursive=True)[0])
import shutil; shutil.copytree(DEB, '/kaggle/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/kaggle/working/deberta_v3_large'
print('part two', N14B, N14BC['keep'], N14BC['layers'], N14BC.get('eos_id'), Q25_14, Q25_14C['keep'], Q25_14C['layers'], DEB)
AQ8V5 = [adir(f'q8v5_f{k}') for k in range(5)] + [adir('q8v5')]
print('q8v5', AQ8V5)
AQ14V5 = [adir(f'q14v5_f{k}') for k in range(5)] + [adir('q14v5')]
ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]
print('v5', AQ14V5, ALL8V5)
