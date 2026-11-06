"""Describe the existing Drive layout; does not contact or modify Drive."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / 'cloud/who-am-i-v32-colab/reference/cli_input_inventory.json'
LIB = 'Model-Library/'
BASE = LIB + '01-Base-Models/'
ADAPT = LIB + '02-Adapters-LoRA/Name-That-LLM/'
ENC = LIB + '03-Encoders-Classifiers/Name-That-LLM/'
SNAP = LIB + '04-Competition-Snapshots/Name-That-LLM/'

renamed = {
 'qwen-lm/qwen-3/transformers/8b/1': BASE+'Qwen-Qwen3-8B-Posttrained-BF16',
 'qwen-lm/qwen-3/transformers/4b-base/1': BASE+'Qwen-Qwen3-4B-Base-BF16',
 'qwen-lm/qwen-3/transformers/8b-base/1': BASE+'Qwen-Qwen3-8B-Base-BF16',
 'qwen-lm/qwen-3/transformers/14b-base/1': BASE+'Qwen-Qwen3-14B-Base-BF16',
 'qwen-lm/qwen-3/transformers/14b/1': BASE+'Qwen-Qwen3-14B-Posttrained-BF16',
 'qwen-lm/qwen2.5/transformers/7b/1': BASE+'Qwen-Qwen2.5-7B-Base-BF16',
 'qwen-lm/qwen2.5/transformers/14b/1': BASE+'Qwen-Qwen2.5-14B-Base-BF16',
 'zhjianduan/llama31-8b-base-verified': BASE+'NousResearch-Meta-Llama-3.1-8B-Base-BF16-rev1f47e50',
 'dhruvtibarewal/nlm-l24-adapters': ADAPT+'NTL-Dhruv-L24-v7-Adapters',
 'dhruvtibarewal/nlm-l24-adapters-v5': ADAPT+'NTL-Dhruv-L24-v5-Adapters',
 'zbigniewkirchner/ntl-qwen3-8b-lora-claude-head': ADAPT+'NTL-Zbigniew-Qwen3-8B-ClaudeHead-LoRA-v1',
 'nishantdahal/ntl-two-stage-lora': ADAPT+'NTL-Nishant-TwoStageLoRA-v1',
 'josephadamski91/ntl-atlas-qwen3-8b-v1': ADAPT+'NTL-Joseph-ATLAS-Qwen3-8B-v1-dataset-v3',
 'pjmathematician/ultrabert-380m-ntl': ENC+'NTL-UltraBERT-380M-5Fold-v1',
}

partial = {
 'nishantdahal/ntl-handwriting-blend': [
  (ADAPT+'NTL-Nishant-HandwritingBlend-Qwen3-14B-LoRA', 'lora_qwen3_14b'),
  (ADAPT+'NTL-Nishant-HandwritingBlend-Qwen2.5-14B-LoRA', 'lora_qwen25_14b'),
  (ENC+'NTL-Nishant-Handwriting-DeBERTa-v3-Large', 'deberta_v3_large')],
 'lucasboesen/name-that-llm-glm4-9b-trunc25': [(SNAP+'NTL-Lucas-GLM4-9B-Trunc25-Snapshot', 'glm4-9b-trunc25')],
 'lucasboesen/name-that-llm-extract-encoders2': [(ENC+'NTL-Lucas-DeBERTa-v3-Large-Encoder', 'deberta-v3-large')],
 'lucasboesen/name-that-llm-electra-large-weights': [(ENC+'NTL-Lucas-ELECTRA-Large-Discriminator', 'electra-large-discriminator')],
 'lucasboesen/name-that-llm-desklib-weights': [(ENC+'NTL-Lucas-Desklib-Detector', 'desklib-detector')],
 'lucasboesen/name-that-llm-train-lora24-s7': [(ADAPT+'NTL-Lucas-LoRA24-S7-v1', 'lora_out')],
 'vnton37nguyn/ntl-train-lora-qwen25-instyle': [(ADAPT+'NTL-Vnton-Qwen2.5-InStyle-LoRA-v2', 'lora_out')],
 'lucasboesen/name-that-llm-train-tokattn': [(ENC+'NTL-Lucas-TokenAttention-Classifier-Heads/tokattn_final.pt', 'tokattn_final.pt')],
 'lucasboesen/name-that-llm-train-tokattn2': [
  (ENC+'NTL-Lucas-TokenAttention-Classifier-Heads/tokattn_debl23_final.pt', 'tokattn_debl23_final.pt'),
  (ENC+'NTL-Lucas-TokenAttention-Classifier-Heads/tokattn_q8l22_final.pt', 'tokattn_q8l22_final.pt')],
}

inputs = []
for item in json.loads(OLD.read_text(encoding='utf-8')):
    ref, kind = item['ref'], item['kind']
    target = {'dataset':'datasets/', 'notebook':'notebooks/', 'model':'models/', 'competition':'competitions/'}[kind] + ref
    entry = {'kind':kind, 'ref':ref, 'version':item['version'], 'target':target,
             'legacy':item['path'].split('/MyDrive/Kaggle/', 1)[1], 'relocations':[]}
    if ref in renamed:
        entry['relocations'] = [{'source':renamed[ref], 'destination':'', 'replace_root':True}]
    elif ref in partial:
        entry['relocations'] = [{'source':s, 'destination':d} for s,d in partial[ref]]
    inputs.append(entry)
assert len(inputs) == 31

required = []
def require(name, pattern, minimum=1):
    required.append({'name':name, 'pattern':pattern, 'minimum':minimum})
for name in ('train.csv','test.csv','sample_submission.parquet'):
    require('比赛数据 '+name, 'competitions/name-that-llm/'+name)
for name in ('nlm_matched_clean.parquet','nlm_pool_v5.parquet','ntl_extra_passages_claude.csv',
             'ntl_generations.csv','gen_passages.csv','ntl_proxy_generations.csv','passages_clean.csv'):
    require(name, '**/'+name)
for prefix, folds in [('ts_q3b8',range(5)),('ts_q3b14',(0,2,4)),('ts_llama8',(0,2,4)),
                      ('ts8_s7',(0,2,4)),('q8v5',range(5)),('q14v5',range(5)),('ll8v5',range(5))]:
    for name in [prefix]+[f'{prefix}_f{i}' for i in folds]:
        require(name+' 分类头', '**/'+name+'/heads.pt')
        require(name+' LoRA 权重', '**/'+name+'/**/adapter_model.safetensors')
        require(name+' LoRA 配置', '**/'+name+'/**/adapter_config.json')
require('Qwen2.5 7B 分类头', '**/ts_q25_7b/heads.pt')
require('Qwen2.5 7B LoRA', '**/ts_q25_7b/**/adapter_model.safetensors')
require('公共 Gaussian 头', '**/heads/q3_8bx.npz')
for name in ('lora8b_v1','lora14b_v1','lora_qwen3_14b','lora_qwen25_14b'):
    require(name+' 配置', '**/'+name+'/meta.json')
    require(name+' LoRA', '**/'+name+'/adapter/adapter_model.safetensors')
    require(name+' 分类头', '**/'+name+'/heads.safetensors')
require('ATLAS 配置', '**/atlas_bundle.json')
require('DeBERTa 分类器', '**/deberta_v3_large/head.safetensors')
require('DeBERTa 分类器配置', '**/deberta_v3_large/meta.json')
require('DeBERTa token attention', '**/tokattn_debl23_final.pt')
for slug in ('name-that-llm-train-lora24-s7','ntl-train-lora-qwen25-instyle'):
    for file in ('COMPLETE.json','norms.safetensors','readouts.npz','valid_features.npz','adapter/adapter_model.safetensors'):
        require(slug+' '+file, '**/'+slug+'/lora_out/job_final/'+file)
for file in ('COMPLETE.json','heads.safetensors','readouts.npz','anchor_features.npz','adapter/adapter_model.safetensors'):
    require('Claude LoRA '+file, '**/ntl-qwen3-8b-lora-claude-head/final/'+file)

model_patterns = ['models/**/config.json', 'datasets/zhjianduan/llama31-8b-base-verified/config.json',
 '**/glm4-9b-trunc25/config.json', '**/deberta-v3-large/config.json', '**/electra-large-discriminator/config.json',
 '**/desklib-detector/config.json','**/deberta_v3_large/encoder/config.json','**/ultrabert-380m-ntl/fold*/config.json']
for pattern in model_patterns:
    require('底座/编码器配置', pattern, 7 if pattern.startswith('models/') else 5 if '/fold*' in pattern else 1)

old_lock = json.loads((PROJECT/'cloud/who-am-i-v32-colab/input-lock.json').read_text(encoding='utf-8'))
hashes = []
for name,digest in old_lock['pinned_csv_sha256'].items():
    slug,file=name.split('/',1)
    parent=next(x['target'] for x in inputs if x['ref'].endswith('/'+slug))
    hashes.append({'path':parent+'/'+file,'sha256':digest})
manifest = {'schema_version':1,'recorded_date':'2026-11-01','inputs':inputs,'required':required,
 'model_config_patterns':model_patterns,'small_file_hashes':hashes,
 'llama_shard_bytes':{'model-00001-of-00004.safetensors':4976698672,'model-00002-of-00004.safetensors':4999802720,
                      'model-00003-of-00004.safetensors':4915916176,'model-00004-of-00004.safetensors':1168138808},
 'note':'Use existing Drive cache versions. Byte identity against every input of EXP-004 has not been established.'}
(HERE/'drive-inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('manifest:',len(inputs),'input groups;',len(required),'required-file checks')
