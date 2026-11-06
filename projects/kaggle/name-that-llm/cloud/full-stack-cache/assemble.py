"""Stage the tested observer dependencies beside generated frozen-source copies.

This prepares local engineering code only. It never starts a notebook/model.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import instrument

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    baseline=ROOT/'reviews/own-scheme-readiness-20261030/baseline-v2.ipynb'
    assert sha(baseline)=='6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8'
    generated=HERE/'generated'
    receipt=json.loads((generated/'instrumentation-receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='GENERATED_NOT_EXECUTED'
    assert receipt['baseline_sha256']==sha(baseline)
    assert receipt['instrument_sha256']==sha(HERE/'instrument.py')
    components=receipt['components']
    assert len(components)==2 and {x['component'] for x in components}=={'main','comp'}
    for item in components:
        filename,expected=instrument.EXPECTED[item['component']]
        original=instrument.SOURCE_DIR/filename
        assert item['source_filename']==filename and item['source_sha256']==expected==sha(original)
        actual=generated/filename
        assert actual.is_file() and sha(actual)==item['generated_sha256'],'Generated source differs: '+filename
        regenerated,_=instrument.instrument_source(original.read_bytes(),item['component'])
        assert actual.read_bytes()==regenerated.encode('utf-8'),'Generated source no longer matches instrumenter'
        instrument.verify_ast_equivalence(original.read_text(encoding='utf-8'),actual.read_text(encoding='utf-8'))
    notebook=json.loads(baseline.read_text(encoding='utf-8'))
    cell=notebook['cells'][4]['source'];cell=cell if isinstance(cell,str) else ''.join(cell)
    assert cell.startswith('%%writefile gauss.py\n')
    files={'gauss.py':cell.split('\n',1)[1].encode('utf-8')}
    for name in ['full_stack_cache.py','full_stack_observer.py','full_stack_replay.py']:
        files[name]=(ROOT/'src'/name).read_bytes()
    for name,data in files.items():
        path=generated/name
        assert not path.exists() or path.read_bytes()==data,'Existing package file differs: '+name
        if not path.exists():
            with path.open('xb') as stream:stream.write(data)
        compile(data,name,'exec')
    context={'schema_version':1,'run_id':'REQUIRES_NEW_CAPTURE_RUN_ID','scope':'public_sample_engineering',
        'required_events':{'main':['main_members','comp_status','main_probability','pre_atlas',
            'atlas_requested','atlas_returned','pre_sinkhorn','postprocessing_base','typography','component_end'],
            'comp':['member','style_margin','style_routed','comp_before_gates','component_end']},
        'launch_ready':False,'requires_before_model_launch':['frozen actual attachment/file hashes',
            'expected member inventory and declared optional omissions','selected evaluation input/scope',
            'CPU/GPU/time budget and current runtime state','outer notebook writes must precede instrumentation',
            'explicit removal of FAST_COMMIT placeholder shortcut only for authorized real capture']}
    context.update(expected_main_members=None,expected_comp_members=None)
    path=HERE/'capture-context-template.json'
    path.write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'schema_version':1,'experiment':'EXP-028','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'LOCAL_ENGINEERING_PACKAGE','model_predictions_generated':False,'cloud_launch_ready':False,
        'baseline_sha256':sha(baseline),'context_template_sha256':sha(path),
        'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
            for p in sorted(generated.iterdir()) if p.is_file()],
        'note':'This package exports/replays scores when run with valid bindings; it is not a completed real full-stack prediction cache.'}
    with (HERE/'package-manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':manifest['status'],'files':len(manifest['files']),'model_predictions_generated':False}))

if __name__=='__main__':main()
