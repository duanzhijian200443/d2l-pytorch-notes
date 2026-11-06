import hashlib,json,pathlib,runpy,sys,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
matches=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()=='9117d147deb4c2ab27a6054655970445095e638bc11a2d506310bf73170f4855']
assert len(matches)==1, 'Expected exactly one hash-pinned protocol'
target=matches[0].parent
protocol=json.loads(matches[0].read_text())
for name,digest in protocol['input_hashes'].items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'guard.py'),run_name='__main__')
