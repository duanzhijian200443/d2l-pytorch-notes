"""Independent set-based result checks, optional new receipt; no model execution."""
import argparse
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


def hashed(value):
    key = ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())
    return hashlib.sha256(key.encode('utf8')).hexdigest()


def main(output=None):
    if output and (EXP / output).exists():
        raise FileExistsError('Verification receipt already exists; no overwrite')
    cfg = json.loads((EXP / 'config.json').read_text(encoding='utf8'))
    met = ROOT / cfg['output_dir']
    summary = json.loads((met / 'summary.json').read_text(encoding='utf8'))
    def table(name):
        p = ROOT / cfg['inputs'][name]['path']
        if p.suffix == '.parquet':
            return pq.read_table(p).to_pandas()
        return pd.read_csv(p, keep_default_na=False)
    pool = table('open_pool')
    hashes = [hashed(t) for t in pool.text]
    pool_set = set(hashes)
    main_set = set()
    for name in ['official', 'matched', 'pool_v5']:
        main_set.update(map(hashed, table(name).text))
    runtime_set = main_set | set(map(hashed, table('a1').text)) | set(map(hashed, table('seyit').text))
    comp_index = pq.read_table(met / 'shared-extra10469-index.parquet').to_pandas()
    runtime_set.update(comp_index.loc[comp_index.model.str.startswith(('gpt', 'gemini', 'glm')), 'normalized_text_sha256'])
    raw_set = set()
    for name in ['official', 'a1', 'matched', 'pool_v5', 'pool_v2', 'seyit', 'lucas', 'abdullah', 'dophuchung', 'prvs']:
        raw_set.update(map(hashed, table(name).text))
    flags = pq.read_table(met / 'row-risk-index.parquet').to_pandas()
    assert flags.id.tolist() == pool.id.tolist()
    assert flags.normalized_text_sha256.tolist() == hashes
    assert len(main_set) == summary['main_runtime_rows'] == 123828
    assert len(pool_set & runtime_set) == summary['risk_counts']['in_a1_runtime_union'] == 97870
    assert len(pool_set - runtime_set) == summary['risk_counts']['novel_text_to_audited_runtime'] == 92718
    assert len(pool_set - raw_set) == summary['risk_counts']['novel_text_to_available_raw'] == 91481
    assert len(pool_set) == len(pool) == 190588
    assert flags.approved_for_training.sum() == 0
    assert flags.independently_verified_served_model.sum() == 0
    assert (flags.higher_traceability_increment_screen_only & flags.in_a1_runtime_union).sum() == 0
    assert (flags.new_recovered_prompt_increment_screen_only & flags.known_a1_runtime_prompt_link).sum() == 0
    for name, item in summary['candidate_inventories'].items():
        c = pd.read_csv(met / (name + '-candidate-index.csv'))
        assert len(c) == item['rows'] and c.id.is_unique
        assert c.approved_for_training.sum() == 0
    provider = {'claude': 'anthropic', 'gemini': 'google', 'gpt': 'openai', 'glm': 'zai', 'kimi': 'moonshot', 'qwen3.8': 'qwen'}
    mismatches = [i for i, (m, lab) in enumerate(zip(pool.model, pool.lab)) if provider.get(m.split('-')[0]) != lab]
    assert not mismatches
    receipt = {'status': 'PASS_SECOND_IMPLEMENTATION_CHECKS', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'method': 'PyArrow read plus Python normalized-text set unions; not another training/OOF evaluation',
        'main_unique_texts': len(main_set), 'runtime_pool_intersection': len(pool_set & runtime_set),
        'runtime_novel_text_upper_bound': len(pool_set - runtime_set), 'available_raw_novel_upper_bound': len(pool_set - raw_set),
        'all_row_ids_and_hashes_match': True, 'candidate_ids_validated': True,
        'provider_label_disagreement_rows': len(mismatches),
        'author_similarity_at_or_above_rounded_0_3': pool.loc[pool.max_cos_train.ge(.3), ['id', 'max_cos_train']].to_dict('records'),
        'author_similarity_boundary_note': '3 stored rounded values equal0.300; no value exceeds0.3. Cannot infer original unrounded values or certify the card claim strictly below0.3.',
        'source_file_missing_by_source': pool.loc[pool.source_file.eq('')].source_dataset.value_counts().to_dict(),
        'no_training_or_remote_mutation': True, 'strict_oof_approved': False}
    if output:
        path = (EXP / output).resolve()
        if not path.is_relative_to(EXP):
            raise ValueError('Output receipt must stay in EXP050')
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    main(p.parse_args().output)
