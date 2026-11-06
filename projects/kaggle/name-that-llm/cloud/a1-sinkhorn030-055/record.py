"""Write EXP055's private save/run and quick-commit records. No network."""
import argparse
from pathlib import Path
import control as c


def entry(path):
    return {'path':path.relative_to(c.ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':c.base.sha(path)}


def write_config():
    manifest=c.EXP/'manifest.json'
    assert c.base.read(manifest)['freeze_status']=='DRAFT'
    state=c.base.state()
    assert state['run_version']==2 and state['save_request_count']==state['run_request_count']==1
    config={'schema_version':1,'experiment_id':'EXP-055','baseline_id':'EXP-046',
        'candidate_preparation':'EXP-049','notebook_receipt_experiment_id':'EXP-049',
        'hypothesis':'A1 with the main Sinkhorn weight changed from 0.45 to 0.30 is compared to original EXP046 A1 on the competition leaderboard.',
        'model':{'name':'A1 full ATLAS20','parameters':{'sinkhorn_control':.45,'sinkhorn_candidate':.30,'sinkhorn_target_mix':.35,
            'atlas_weight':.2,'comp_weight':.15,'typography_scale':1.5,'self_generated_rows_added':0}},
        'candidate_notebook_ref':c.REF,'candidate_sha256':c.SOURCE,'metadata_sha256':c.METADATA,
        'environment':{'private':True,'internet':False,'requested_t4_shape':c.base.MACHINE,'docker_image':c.base.IMAGE},
        'input_policy':{'requested_assets':32,'noncompetition_files_expected':528,'baseline_small_sha256':398,
                        'extra_small_sha256':5,'baseline_large_exact_size_only':130},
        'execution_scope':'V1 save-only and V2 public quick-commit; separate submission handoff to parent',
        'submission_owner':'parent controller only','recorded_at':c.base.now()}
    c.base.save(c.EXP/'config.json',config)
    receipts=[c.HERE/'discovery.json',c.HERE/'save-only-intent.json',c.HERE/'save-only-response.json',
              c.HERE/'run-intent.json',c.HERE/'run-response.json']
    receipts += [c.HERE/state['readback_receipts'][str(v)]['path'] for v in (1,2)]
    code=[c.HERE/'control.py',c.HERE/'verify_quick.py',c.ROOT/'cloud/a1-public-stylegram-046/control.py']
    inputs=[c.FROZEN/'kernel/a1.ipynb',c.FROZEN/'kernel/kernel-metadata.json',
            c.ROOT/'experiments/EXP-049-a1-sinkhorn030-ready/manifest.json']
    c.base.save(manifest,{'schema_version':1,'experiment_id':'EXP-055','freeze_status':'FIXED_EXP049_SOURCE_V2_READBACK',
        'recorded_at':c.base.now(),'config':entry(c.EXP/'config.json'),'code':[entry(p) for p in code],
        'inputs':[entry(p) for p in inputs],'remote_save_run_evidence':[entry(p) for p in receipts],
        'candidate_prepared_before_dispatch':True,'manifest_recorded_after_readback':True})

def write_ready():
    ready=c.base.read(c.HERE/'ready-for-submission.json')
    c.base.save(c.EXP/'metrics/quick-commit.json',ready)
    report=f"""# EXP-055 · A1 的 Sinkhorn 0.30 正式候选

在 {ready['verified_at']}，公开快保存验收通过，状态为 **READY_FOR_SUBMISSION**。Kaggle 私有 Notebook V2 的 scriptVersionId 为 **{ready['script_version_id']}**。[核验收据](metrics/quick-commit.json)

候选使用 EXP049 冻结 Notebook。相比 EXP046 的 A1，仅将主命令 Sinkhorn 从 0.45 调为 0.30。Stylegram 使用官方 2699 条与原 public10073 配方，未添加自产 120 条。EXP046 的正式 Public LB 0.663 是比较对象。EXP049 的外部面板诊断另见[原实验](../EXP-049-a1-sinkhorn030-ready/report.md)。

## 公开执行

官方 SDK 查询了 74 个自有 Notebook 和 7 条比赛提交。随后各发送一次 V1 保存请求与 V2 运行请求。两个版本均已固定读回，对比完整 Notebook 规范化 JSON、附件集合、私有状态、断网设置、容器和机器规格。[保存、读回与执行证据](manifest.json)

V2 的 preflight 校验 398 个基线小文件及 5 个额外小文件的 SHA256，对 130 个基线大文件核对尺寸。运行环境确认两张 Tesla T4，A1 合成分数与推理路由核对通过。Kaggle 回传部分 dataset/kernel 引用时省略版本，实际字节依据本次运行的分级校验解释。

公开测试集三条 ID 按顺序输出，预测均为快保存占位类。收据标记 EXP-049、Sinkhorn 0.30、ATLAS 0.20、comp 0.15。submission.parquet SHA256 为 `{ready['submission_sha256']}`。正式提交和隐藏评分由统一提交控制器另行记录，本收据仅核验公开快保存。
"""
    (c.EXP/'report.md').write_text(report,encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('config','ready'))
    if parser.parse_args().phase=='config':write_config()
    else:write_ready()
