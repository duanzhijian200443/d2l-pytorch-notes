import hashlib, pathlib, json, runpy, sys, os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
# Kaggle extracts uploaded ZIP datasets into individual input files.
paths=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()=='3983ceb881750b9b138ee59a3c03ba1acb2dd0f04c2b56bbe96ec682446ce50c']
assert len(paths)==1, 'Expected one hash-pinned protocol'
target=paths[0].parent
protocol=json.loads(paths[0].read_text())
for name,digest in protocol['input_hashes'].items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest, name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'runner.py'),run_name='__main__')
