"""Check public engineering entrypoints against frozen synthetic fixtures only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix='ntl-exp028-cli-') as directory:
        temp=Path(directory)
        for case,code in [('n60_full-r2',0),('n60_partial-r2',2),('null_anchor',1),('wrong_anchor',1)]:
            fixture=HERE/'fixtures'/('n60_partial-r2' if case=='n60_partial-r2' else 'n60_full-r2')
            receipts=fixture/'cli-receipts.json'
            if case.endswith('anchor'):
                values=json.loads(receipts.read_text(encoding='utf-8'))
                values['main']['manifest_sha256']=None if case=='null_anchor' else '0'*64
                receipts=temp/(case+'-receipts.json')
                receipts.write_text(json.dumps(values),encoding='utf-8')
            output=temp/(case+'.json')
            args=[sys.executable,'-B','-X','utf8',str(ROOT/'cloud/full-stack-cache/verify_cache.py'),
                  '--cache-root',str(fixture/'cache'),'--test-csv',str(fixture/'inputs/test.csv'),
                  '--receipts-json',str(receipts),'--inventory-json',str(fixture/'cli-inventory.json'),
                  '--output',str(output)]
            result=subprocess.run(args,cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
            assert result.returncode==code,(case,result.returncode,result.stderr)
            if code in (0,2):
                record=json.loads(output.read_text(encoding='utf-8'))
                assert record['accepted'] is (code==0)
                assert record['input_scope']=='synthetic_math'
                assert len(record['helper_files_sha256'])==3
                (HERE/'fixtures'/case/'cli-verification-r3.json').write_bytes(output.read_bytes())
                status=record['status']
            else:
                assert not output.exists(),'Rejected input created an acceptance report'
                status='REJECTED'
            results.append({'case':case,'returncode':result.returncode,'status':status})

        # Corrupt only a temporary copy. Never modify generated or original sources.
        package=ROOT/'cloud/full-stack-cache'
        sys.path.insert(0,str(package))
        spec=importlib.util.spec_from_file_location('cache_assemble_check',package/'assemble.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        stage=temp/'package';(stage/'generated').mkdir(parents=True)
        for name in ('instrumentation-receipt.json','infer.py','comp_stack.py'):
            (stage/'generated'/name).write_bytes((package/'generated'/name).read_bytes())
        (stage/'instrument.py').write_bytes((package/'instrument.py').read_bytes())
        with (stage/'generated/infer.py').open('ab') as stream:stream.write(b'\n# changed copy\n')
        module.HERE=stage
        try:module.main()
        except AssertionError as error:
            assert 'Generated source differs' in str(error),str(error)
        else:raise AssertionError('Assembler accepted altered generated source')
        assert not (stage/'package-manifest.json').exists()
        assert not (stage/'generated/full_stack_cache.py').exists()
        results.append({'case':'modified_generated_source','status':'REJECTED_BEFORE_STAGING'})

    record={'status':'PASS_ENGINEERING_ENTRYPOINTS','scope':'synthetic_fixtures_and_temporary_tamper_checks',
            'new_model_execution':False,'results':results,
            'code_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),package/'assemble.py',package/'verify_cache.py']}}
    path=HERE/'metrics/entrypoints-verified.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':record['status'],'checks':len(results)}))


if __name__=='__main__':main()
