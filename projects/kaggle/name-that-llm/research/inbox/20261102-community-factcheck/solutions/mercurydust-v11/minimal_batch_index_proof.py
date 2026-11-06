"""Run only the original AST-selected inference loop with synthetic arrays and a recording stub.
No notebook initialization, network, fitting, backbone, GPU, or external data execution.
The corrected loop is a diagnostic comparator, not an upstream code modification.
"""
from pathlib import Path
import ast, copy, hashlib, json
import numpy as np
from scipy import sparse

BASE = Path(__file__).resolve().parent
nb = json.loads((BASE/'source.ipynb').read_text(encoding='utf-8'))
cell = next(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('%%writefile v34_pairwise.py'))
source = cell.split('\n',1)[1]
tree = ast.parse(source)
main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loop = next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='start')
original = ast.Module(body=[copy.deepcopy(loop)],type_ignores=[])

class CorrectGlobalIndex(ast.NodeTransformer):
    changes = 0
    def visit_Subscript(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id in {'Cte','Wte','test_texts'} and isinstance(node.slice,ast.BinOp) and isinstance(node.slice.op,ast.Sub) and isinstance(node.slice.right,ast.Name) and node.slice.right.id=='start':
            node.slice=node.slice.left; self.changes+=1
        return node

fix=CorrectGlobalIndex(); corrected=fix.visit(copy.deepcopy(original)); ast.fix_missing_locations(corrected)
assert fix.changes==5

class RecordingVerifier:
    def __init__(self): self.features=[]
    def predict_proba(self,x):
        self.features.append(x.copy())
        # Fixed synthetic score only to satisfy original loop's interface; no fitted model.
        return np.tile([0.5,0.5],(len(x),1))

def run(tree,rows):
    n=6; R=6
    test=np.tile([1.,0.],(rows,1))
    if rows>128: test[128:]=[0.,1.]
    refs=np.tile([1.,0.],(n,1))
    aprof=np.tile([1.,0.],(n+rows,1)); aprof[n+min(rows,128):]=[0.,1.]
    btest=np.full((rows,4),'first',dtype=object); btest[128:]='later'
    verifier=RecordingVerifier()
    punct=np.zeros((n,11),dtype=np.float32); punct[:,0]=0.5
    env={'np':np,'te':list(range(rows)),'LABELS':['synthetic'],'n':n,'R':R,'scores':np.zeros((rows,1)),
         'Cte':sparse.csr_matrix(test),'Wte':sparse.csr_matrix(test),'Ctr':sparse.csr_matrix(refs),'Wtr':sparse.csr_matrix(refs),
         'idx_by_model':{0:np.arange(n)},'a_prof':aprof,'A_tr':aprof[:n],'b_test':btest,'b_oof':np.full((n,4),'first',dtype=object),
         'test_texts':['one.']*min(rows,128)+['one two three four five!!!']*max(0,rows-128),
         'lengths':np.ones(n,dtype=np.float32),'punct':punct,'verifier':verifier}
    exec(compile(tree,'<AST selected original batch inference loop>','exec'),env)
    return np.concatenate(verifier.features).reshape(rows,R,10)

cases=[]
for rows in [3,128,129,257]:
    bad=run(original,rows); good=run(corrected,rows)
    mask=np.any(~np.isclose(bad,good),axis=(1,2)); changed=np.flatnonzero(mask).tolist()
    assert changed==list(range(128,rows))
    assert np.allclose(bad[:,:,:8][:,:,2:],good[:,:,:8][:,:,2:])
    case={'test_rows':rows,'mismatched_zero_based_rows':changed,'changed_feature_columns':np.flatnonzero(np.any(~np.isclose(bad,good),axis=(0,1))).tolist()}
    if rows>128:
        case['row129_original_first_reference_features']=bad[128,0].tolist()
        case['row129_corrected_first_reference_features']=good[128,0].tolist()
        assert case['changed_feature_columns']==[0,1,8,9]
    cases.append(case)
result={'status':'PASS_REPRODUCED_INDEX_BUG','source_sha256':hashlib.sha256((BASE/'source.ipynb').read_bytes()).hexdigest(),
        'method':'AST-selected original main inference loop; synthetic sparse matrices; recording verifier stub; no fitting',
        'comparator_replacements':fix.changes,'features':['char_cosine','word_cosine','A_cosine','A_absdiff','same_task','same_domain','same_structure','same_explicitness','length_ratio','punctuation_similarity'],
        'cases':cases,'claims_not_tested':['real model predictions','effect size on leaderboard','full auxiliary pipeline','full notebook execution']}
(BASE/'minimal-proof-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
