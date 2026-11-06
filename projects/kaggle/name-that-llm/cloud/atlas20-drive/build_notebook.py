"""Apply the Drive path adaptation to the user's saved Notebook, once."""
import ast
import hashlib
import json
from pathlib import Path
import pprint

import nbformat

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TARGET = PROJECT / 'notebooks/name-that-llm-atlas20-repro.ipynb'
BACKUP = PROJECT / 'reviews/atlas20-drive-20261101/before-drive-adaptation.ipynb'
RECEIPT = BACKUP.parent / 'adaptation-validation.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    if TARGET.read_bytes() != BACKUP.read_bytes():
        raise RuntimeError('Notebook changed since backup; preserve the new edits before rebuilding.')
    before = nbformat.read(BACKUP, as_version=4)
    manifest = json.loads((HERE/'drive-inputs.json').read_text(encoding='utf-8'))
    helper = (HERE/'drive_inputs.py').read_text(encoding='utf-8')
    ast.parse(helper)
    introduction = nbformat.v4.new_markdown_cell('''# ATLAS20 · Colab / Google Drive 输入适配

使用已经挂载的 `MyDrive/Kaggle/Model-Library` 和 `name-that-llm/v32-colab/cli-cache`。
按下面顺序运行：**0.1 挂载 → 0.2 输入检查 → 1. Inputs**。挂载/检查阶段无需 GPU，不下载模型，也不安装依赖。

当前 `FULL_RUN = False`，运行全部单元格也只做准备，不会生成占位预测或启动大模型计算。
只有输入检查通过、GPU 与依赖就绪且准备执行完整实验时，才将它改为 `True` 并从头运行。
完整实验结束后仍需保存产物并释放 Colab 运行时；本次仅验证路径适配，尚未执行完整 GPU 流程。

模型成员、融合参数和算法沿用当前 ATLAS20 文件；权重使用 Drive 中已有版本。
历史 0.662 属于 Kaggle 正式提交，并非此 Colab 文件的成绩。尚未逐字节证明所有 Drive 输入都与那次提交一致。
下方保留作者原说明与出处。''', id='atlas-drive-intro')
    mounting = nbformat.v4.new_code_cell('''# 0.1 挂载 Google Drive（复用你已完成的挂载）
from pathlib import Path
from google.colab import drive

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
DRIVE_KAGGLE = Path('/content/drive/MyDrive/Kaggle')
MODEL_ROOT = DRIVE_KAGGLE / 'Model-Library/01-Base-Models'
QWEN14B_DIR = MODEL_ROOT / 'Qwen-Qwen3-14B-Base-BF16'
assert (QWEN14B_DIR / 'config.json').is_file(), f'未找到模型库: {QWEN14B_DIR}'
print('Drive 可用；Qwen3-14B 分片数:', len(list(QWEN14B_DIR.glob('*.safetensors'))))
''', id='baf8d2ab')
    settings = nbformat.v4.new_code_cell('''# 0.2 先检查全部输入；不申请 GPU、不加载权重
FULL_RUN = False
INPUT_ROOT = Path('/content/atlas20-drive/input')
WORK_ROOT = Path('/content/atlas20-drive/working')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
'''+ '\nDRIVE_MANIFEST = ' + pprint.pformat(manifest, width=110, sort_dicts=False) + '\n\n' + helper + '''
INPUT_REPORT = prepare_drive_inputs(
    DRIVE_MANIFEST, DRIVE_KAGGLE, INPUT_ROOT, WORK_ROOT / 'drive_input_preflight.json'
)
os.chdir(WORK_ROOT)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
print(f"输入检查通过: {INPUT_REPORT['input_groups']} 组，{INPUT_REPORT['linked_files']} 个文件别名")
print(f"当前 test.csv: {INPUT_REPORT['test_rows']} 行；没有执行推理。")
print('完整模型字节哈希未全部重算；这次结果不是榜单成绩。')
''', id='atlas-drive-preflight')
    settings.metadata['jupyter'] = {'source_hidden': True}
    nb = nbformat.from_dict(before)
    cells = []
    removed_mounts = []
    script_hashes = {}
    for cell in nb.cells:
        source = cell.source
        if cell.cell_type == 'code' and 'drive.mount(' in source:
            # Only consolidate the three known mount/check cells, not arbitrary user code.
            if cell.id not in {'6d375f49','f5438f97','baf8d2ab'}:
                raise RuntimeError(f'Unexpected user mount cell {cell.id}; inspect before replacing.')
            removed_mounts.append(cell.id)
            continue
        if cell.cell_type == 'code':
            original = source
            source = source.replace('/kaggle/input', '/content/atlas20-drive/input')
            source = source.replace('/kaggle/working', '/content/atlas20-drive/working')
            if cell.id == 'ee28391a':
                start = "whl = glob.glob('/content/atlas20-drive/input/**/bitsandbytes-*.whl', recursive=True)\nif whl: print(subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)"
                if start not in source: raise RuntimeError('Unexpected dependency setup')
                source = source.replace(start, '''assert 'INPUT_REPORT' in globals() and INPUT_REPORT['status'] == 'READY', '请先运行顶部挂载与输入检查'
if FULL_RUN:
    import importlib.util
    required_packages = ['torch','transformers','peft','accelerate','safetensors','numpy','pandas',
                         'scipy','sklearn','lightgbm','pyarrow','sentencepiece']
    missing_packages = [p for p in required_packages if importlib.util.find_spec(p) is None]
    assert not missing_packages, '缺少依赖，先配置环境: ' + ', '.join(missing_packages)
    import torch
    assert torch.cuda.is_available(), '完整推理需要连接 GPU；当前检查模式不需要 GPU'
    print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__)
    # Exercise CUDA before reading any large model files.
    print('CUDA probe:', (torch.ones(2, device='cuda') + 1).tolist())''')
                old_fast = "FAST_COMMIT = len(_pd.read_csv(DATA + '/test.csv')) < 50   # fork: the 3-row commit run skips the heavy cells; the hidden rerun runs everything"
                if old_fast not in source: raise RuntimeError('Unexpected FAST_COMMIT line')
                source = source.replace(old_fast, "FAST_COMMIT = not FULL_RUN  # Colab: explicit input-check mode; never infer success from test size")
                old_llama = "shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True); BL = '/tmp/llama-3.1/8b'   # fork: see the intro"
                source = source.replace(old_llama, "if FULL_RUN:\n    shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True); BL = '/tmp/llama-3.1/8b'")
                old_deb = "import shutil; shutil.copytree(DEB, '/content/atlas20-drive/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/content/atlas20-drive/working/deberta_v3_large'"
                source = source.replace(old_deb, "if FULL_RUN:\n    shutil.copytree(DEB, '/content/atlas20-drive/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/content/atlas20-drive/working/deberta_v3_large'")
            elif cell.id == '76257117':
                source = source.replace("    !python comp_stack.py 2>&1 | grep -vE 'Warning|warn'", "    subprocess.run([sys.executable, 'comp_stack.py'], check=True)")
            elif cell.id == 'b0c390ab':
                prefix = source.split('if FAST_COMMIT:',1)[0]
                source = prefix + '''if FAST_COMMIT:
    print('输入检查模式结束：未执行完整推理，没有生成 submission.parquet。')
else:
    subprocess.run(cmd, check=True)
    submission = pd.read_parquet('/content/atlas20-drive/working/submission.parquet')
    assert len(submission) == len(pd.read_csv(os.path.join(DATA, 'test.csv')))
    assert set(submission.columns) == {'id', 'model'} and submission.id.is_unique and submission.model.notna().all()
    print(submission.head())
    print('这是当前 test.csv 的预测；公开样例不能验证榜单分数。')
'''
            cell.source = source
            cell.outputs = []
            cell.execution_count = None
            for key in ('execution','papermill'):
                cell.metadata.pop(key, None)
            if source.startswith('%%writefile'):
                name = source.splitlines()[0].split()[-1]
                reverted = source.replace('/content/atlas20-drive/input','/kaggle/input').replace('/content/atlas20-drive/working','/kaggle/working')
                assert reverted == original, f'Algorithm changed in {name}'
                script_hashes[name] = {'before':digest(original.encode()),'after':digest(source.encode()),'path_only_change':True}
                ast.parse('\n'.join(source.splitlines()[1:]))
            else:
                ast.parse(source)
        elif cell.id == '363f96c4':
            cell.source = '## 1. Inputs\n从前面建立的 Drive 文件别名定位全部输入。先检查路径；仅在 `FULL_RUN=True` 时复制 Llama 和 DeBERTa 到运行时临时磁盘并检查 GPU。'
        elif cell.id == '03e73322':
            cell.source += '\n\nColab 适配：`FULL_RUN=False` 时只检查，不生成预测；`True` 时在当前 `test.csv` 上执行原集成。'
        cell.metadata.pop('papermill', None)
        cells.append(cell)
    if len(removed_mounts) != 3: raise RuntimeError('Expected the three saved mount cells')
    # Setup comes before the original Inputs cell; original attribution remains intact.
    nb.cells = [introduction, mounting, settings] + cells
    nb.metadata.pop('papermill',None)
    nb.metadata['atlas20_drive'] = {'version':1,'full_run_default':False,
        'source_backup':'reviews/atlas20-drive-20261101/before-drive-adaptation.ipynb',
        'drive_manifest_sha256':digest((HERE/'drive-inputs.json').read_bytes())}
    for cell in (mounting,settings): ast.parse(cell.source)
    nbformat.validate(nb)
    nbformat.write(nb,TARGET)
    receipt = {'validation_date':'2026-11-01','target':TARGET.relative_to(PROJECT).as_posix(),
        'backup_sha256':digest(BACKUP.read_bytes()),'adapted_sha256':digest(TARGET.read_bytes()),
        'notebook_format':'PASS','python_syntax':'PASS','algorithm_scripts':script_hashes,
        'input_groups':len(manifest['inputs']),'required_file_rules':len(manifest['required']),
        'gpu_started':False,'full_inference_executed':False,'remote_input_preflight_executed':False,
        'note':'Runtime checks must still be executed in the mounted Colab kernel. Cached Drive inputs are not proven byte-identical to all EXP-004 inputs.'}
    RECEIPT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    build()
