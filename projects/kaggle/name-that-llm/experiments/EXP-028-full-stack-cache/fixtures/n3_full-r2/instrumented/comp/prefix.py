# Harness-selected original math; model paths are excluded.
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}
W = {'stylegram': 1.516, 'stylometry_lgb': 0.063, 'ultrabert': 0.095, 'q3_8b': 0.0, 'q25_7b': 0.0, 'q3_4b': 0.0, 'glm9b': 0.0, 'debv3l': 0.05, 'electl': 0.032, 'desk': 0.132, 'lora24_public': 0.177, 'lora24_seed7': 0.113, 'lora24_claude': 0.177, 'lora20_q25_instyle': 0.17, 'tokattn_q35': 0.153, 'tokattn_deb': 0.185}
TWIN_W = {'q3_8b': 1.5 * 0.219, 'q25_7b': 1.5 * 0.141, 'q3_4b': 1.5 * 0.083, 'glm9b': 1.5 * 0.11, 'debv3l': 0.25}
LORA_TWIN_W = {'lora24_public': 0.15, 'lora24_claude': 0.15}
GATE_LAM = {'claude_style': 1.5, 'claude_lora': 1.0, 'family': 0.25}
GAUSS_HEADS = {'gauss_q3_8b_L22': dict(src='q3_8b', part='L22_mean', d=384, T1=2.351, Tg=2.306, w=TWIN_W['q3_8b']), 'gauss_glm9b_L8': dict(src='glm9b', part='L8_mean', d=384, T1=2.722, Tg=2.596, w=TWIN_W['glm9b']), 'gauss_q3_4b_L4': dict(src='q3_4b', part='L4_mean', d=384, T1=2.725, Tg=2.578, w=TWIN_W['q3_4b']), 'gauss_debv3l_L24': dict(src='debv3l', part='L24_mean', d=384, T1=3.112, Tg=3.069, w=TWIN_W['debv3l'])}
LOGIT = np.zeros((len(TE), K))
from full_stack_observer import FullStackObserver as _cache_observer
_cache = _cache_observer.from_env('comp', test['id'].astype(str).tolist(), test['text'].tolist(), LABELS, {'source_sha256': 'b6b6a7b8adbe50580c2d0de5a41d0764ae752fe729e1471872a9197ac956f222', 'smoke': SMOKE, 'tiny': TINY, 'fake_test': os.environ.get('NTL_FAKE_TEST'), 'drop_q35': os.environ.get('NTL_DROP_Q35'), 'data_dir': os.environ.get('NTL_DATA') if SMOKE else str(DATA_DIR), 'train_rows': len(train)})

def add_member(key, prob, rows=None, weight=None):
    """Add weight * log(prob) to the blend (all rows, or the given row subset)."""
    w = W[key] if weight is None else weight
    lp = np.log(np.clip(prob, 1e-09, 1.0))
    _cache.capture('member', key=key, probability=prob, log_probability=lp, weight=w, rows=rows)
    if rows is None:
        LOGIT[:] += w * lp
    else:
        LOGIT[rows] += w * lp
    print(f'  + {key:20s} weight {w:.3f} on {(len(TE) if rows is None else int(np.sum(rows)))} rows  ({time.time() - T0:.0f}s)', flush=True)

def _np_logsoftmax(z):
    from scipy.special import logsumexp as _lse_np
    z = np.asarray(z, dtype=np.float64)
    return z - _lse_np(z, axis=1, keepdims=True)

def _apply_conditional_family_gate(gate_key, family, lam, scores_are_logprob=False):
    """Use the already-trained family gate only to rerank siblings, preserving total family mass.
    Confidence-aware interpolation avoids overriding high-confidence cross-family decisions."""
    score = GATES.get(gate_key)
    if score is None:
        print(f'  family gate skipped {gate_key}: no fitted scores', flush=True)
        return
    idx = FAMILY_IDX[family]
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (len(TE), len(idx)):
        print(f'  family gate skipped {gate_key}: shape {score.shape} expected {(len(TE), len(idx))}', flush=True)
        return
    if scores_are_logprob:
        gate_lp = _np_logsoftmax(score)
    else:
        score = score - score.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(score * score, axis=1, keepdims=True))
        score = score / np.maximum(scale, 0.25)
        gate_lp = _np_logsoftmax(score)
    base = LOGIT[:, idx].copy()
    base_lp = _np_logsoftmax(base)
    base_p = np.exp(base_lp)
    sp = np.sort(base_p, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    confidence_gate = np.clip((0.45 - margin) / 0.45, 0.0, 1.0)
    strength = min(0.45, float(lam) / (1.0 + float(lam))) * confidence_gate
    mixed_lp = _np_logsoftmax((1.0 - strength[:, None]) * base_lp + strength[:, None] * gate_lp)
    from scipy.special import logsumexp as _lse_np
    family_mass = _lse_np(base, axis=1)
    LOGIT[:, idx] = family_mass[:, None] + mixed_lp
    _cache.capture('family_gate_applied', key=gate_key, family=family, indices=idx, raw_score=GATES[gate_key], gate_log_probability=gate_lp, base=base, strength=strength, family_mass=family_mass, logit=LOGIT)
    print(f'  family gate applied {gate_key}: mean strength={strength.mean():.4f}, active rows={int((strength > 0).sum())}', flush=True)
