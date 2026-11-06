import hashlib,json,pathlib,runpy,sys,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
matches=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()=='d4d2737031fadaa9a20382813ab215a3ee22f680231d017141eeb706117123b2']
assert len(matches)==1, 'Expected exactly one hash-pinned protocol'
target=matches[0].parent
protocol=json.loads(matches[0].read_text())
for name,digest in protocol['input_hashes'].items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'guard.py'),run_name='__main__')
