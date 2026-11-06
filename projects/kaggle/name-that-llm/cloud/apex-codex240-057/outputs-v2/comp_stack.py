"""Complementary frozen-feature stack (public Apache-2.0 notebooks by prvsiyan / lucasboesen / vnton37nguyn / kirilltym,
as embedded in the public 0.6xx ATLAS notebooks). Run as a separate process; writes its raw log-probability blend."""
# ============================================================ 0. configuration
import gc, glob, json, os, pathlib, re, time
import numpy as np, pandas as pd
try:
    import torch
except ImportError:   # local smoke without torch
    torch = None
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SMOKE = os.environ.get('NTL_SMOKE') == '1'   # local code-path check: pseudo features instead of checkpoints, CPU
T0 = time.time()
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}

# ---- test-pool grouping
GROUP_THRESH, GROUP_LAYER, GROUP_MIN, GROUP_MAX = 0.60, 24, 2, 18   # mixed content + GLM-L24 distance, average linkage
STYLE_GROUP_THRESH = 0.90                                           # stylegram's own median-normalised content distance

# ---- blend weights: log-probability multipliers of the 0.563 notebook (its L-BFGS fit on prompt-grouped OOF)
W = {'stylegram': 1.516, 'stylometry_lgb': 0.063, 'ultrabert': 0.095,
     'q3_8b': 0.0, 'q25_7b': 0.0, 'q3_4b': 0.0, 'glm9b': 0.0, 'debv3l': 0.05, 'electl': 0.032, 'desk': 0.132,
     'lora24_public': 0.177, 'lora24_seed7': 0.113, 'lora24_claude': 0.177, 'lora20_q25_instyle': 0.17,
     'tokattn_q35': 0.153, 'tokattn_deb': 0.185}
TWIN_W = {'q3_8b': 1.5 * 0.219, 'q25_7b': 1.5 * 0.141, 'q3_4b': 1.5 * 0.083, 'glm9b': 1.5 * 0.110, 'debv3l': 0.25}
LORA_TWIN_W = {'lora24_public': 0.15, 'lora24_claude': 0.15}
GATE_LAM = {'claude_style': 1.5, 'claude_lora': 1.0, 'family': 0.25}  # used by confidence-aware within-family gates below
EXTRA_MIN_LABELS, EXTRA_SHARE = 12, 0.5

# ---- the addition: shared-prompt Gaussian group heads on four measured views, weight = the centred twin of the same backbone,
#      reduction / temperatures from our production fit (research/gaussian/production_weights_v8g.json, 135 training prompts)
GAUSS_HEADS = {'gauss_q3_8b_L22': dict(src='q3_8b', part='L22_mean', d=384, T1=2.351, Tg=2.306, w=TWIN_W['q3_8b']),
               'gauss_glm9b_L8': dict(src='glm9b', part='L8_mean', d=384, T1=2.722, Tg=2.596, w=TWIN_W['glm9b']),
               'gauss_q3_4b_L4': dict(src='q3_4b', part='L4_mean', d=384, T1=2.725, Tg=2.578, w=TWIN_W['q3_4b']),
               'gauss_debv3l_L24': dict(src='debv3l', part='L24_mean', d=384, T1=3.112, Tg=3.069, w=TWIN_W['debv3l'])}
GAUSS_SETTING = dict(red='wpca', se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)

# ---- member specifications
LM_HEADS = [  # key, model_type, hidden size, parts of the plain head [(layer, pool)], extra parts extracted in the same pass, C
    ('glm9b', 'glm4', 4096, [(8, 'mean'), (3, 'mean')], [(GROUP_LAYER, 'mean')], 0.0025),   # first: its L24 builds the groups
    ('q3_8b', 'qwen3', 4096, [(7, 'mean'), (22, 'mean')], [], 0.0025),
    ('q25_7b', 'qwen2', 3584, [(6, 'mean'), (20, 'mean')], [], 0.0025),
    ('q3_4b', 'qwen3', 2560, [(4, 'style'), (7, 'style')], [(4, 'mean')], 0.0025)]
ENC_HEADS = [  # key, input directory name, parts, C
    ('debv3l', 'deberta-v3-large', [(24, 'mean'), (6, 'mean')], 0.005),
    ('electl', 'electra-large-discriminator', [(21, 'mean'), (24, 'style')], 0.005),
    ('desk', 'desklib-detector', [(23, 'mean'), (23, 'style')], 0.005)]
LORA_RUNS = [  # (trunk model_type, hidden, kept layers), pooled layers, [(key, checkpoint marker, norms file, anchor features file)]
    (('qwen3', 4096, 24), [2, 4, 6, 8, 12, 16, 20, 24], [
        ('lora24_public', '**/lora24/final/COMPLETE.json', 'heads.safetensors', 'anchor_features.npz'),
        ('lora24_seed7', '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json', 'norms.safetensors', 'valid_features.npz'),
        ('lora24_claude', '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json', 'heads.safetensors', 'anchor_features.npz')]),
    (('qwen2', 3584, 20), [2, 4, 6, 8, 10, 12, 16, 20], [
        ('lora20_q25_instyle', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', 'norms.safetensors', 'valid_features.npz')])]
LORA_MAX_LEN, LORA_BS = 447, 8
TOKATTN_HEADS = [('tokattn_q35', 'tokattn_final.pt', 'qwen3_5'), ('tokattn_deb', 'tokattn_debl23_final.pt', 'deberta')]
if os.environ.get('NTL_DROP_Q35') == '1':   # Qwen3.5-9B token-attention member: 1.7-2.7 h fixed cost on Kaggle, weight 0.153 of ~3.3
    TOKATTN_HEADS = [t for t in TOKATTN_HEADS if t[0] != 'tokattn_q35']

# ---- data
if SMOKE:
    _all = pd.read_csv(os.environ['NTL_DATA'] + '/train.csv')
    _p = _all['prompt_id'].drop_duplicates().sample(frac=1, random_state=0).tolist()
    train = _all[_all['prompt_id'].isin(_p[:30])].reset_index(drop=True)
    test = _all[_all['prompt_id'].isin(_p[30:42])].sample(frac=1, random_state=1).reset_index(drop=True)
    test = pd.concat([test[test['prompt_id'] == p].head(k) for p, k in zip(_p[30:42], [1, 1, 2, 2, 2, 3, 3, 4, 6, 9, 18, 18])]).reset_index(drop=True)
    if os.environ.get('NTL_TINY') == '1':
        test = test.head(3).reset_index(drop=True)
    SMOKE_Y = test['model'].map(LAB2I).to_numpy(dtype=int)
else:
    DATA_DIR = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'), pathlib.Path('/kaggle/input/name-that-llm')]
                    if (p / 'train.csv').exists())
    train, test = pd.read_csv(DATA_DIR / 'train.csv'), pd.read_csv(DATA_DIR / 'test.csv')
    if os.environ.get('NTL_FAKE_TEST'):   # timing probe: pool passages as a stand-in test (same rows/seed as infer.py --fake_test)
        test = pd.read_parquet(os.environ['NTL_POOL']).sample(int(os.environ['NTL_FAKE_TEST']), random_state=0)[['text']].reset_index(drop=True)
        test['id'] = [f'fake{i}' for i in range(len(test))]
test['text'] = test['text'].fillna('').astype(str)
TR, TE = train['text'].tolist(), test['text'].tolist()
y = train['model'].map(LAB2I).to_numpy(dtype=int)
PROMPT = train['prompt_id'].to_numpy(dtype=object)
TINY = len(TE) < 50   # the interactive 3-row test: exercise the group paths on training rows so the hidden rerun cannot hit untested code
DEV = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
print(f'train {train.shape}  test {test.shape}  gpus {torch.cuda.device_count() if torch is not None else 0}  smoke {SMOKE}', flush=True)

LOGIT = np.zeros((len(TE), K))           # the running log-probability blend


def add_member(key, prob, rows=None, weight=None):
    """Add weight * log(prob) to the blend (all rows, or the given row subset)."""
    w = W[key] if weight is None else weight
    lp = np.log(np.clip(prob, 1e-9, 1.0))
    if rows is None:
        LOGIT[:] += w * lp
    else:
        LOGIT[rows] += w * lp
    print(f'  + {key:20s} weight {w:.3f} on {len(TE) if rows is None else int(np.sum(rows))} rows  ({time.time() - T0:.0f}s)', flush=True)


def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)


def full_prob(P_part, classes):
    """Expand a (n, len(classes)) probability block to (n, K) on the global label order."""
    P = np.zeros((len(P_part), K))
    P[:, classes] = P_part
    return P


def group_centre_dense(X, groups):
    X = np.asarray(X, dtype=np.float32).copy()
    for g in np.unique(groups):
        i = groups == g
        X[i] -= X[i].mean(0)
    return X


def self_test_rows():
    """Training rows of the first three prompts, grouped by their true prompt (used only when the test file is tiny)."""
    idx = np.where(np.isin(PROMPT, pd.unique(PROMPT)[:3]))[0]
    return idx, np.unique(PROMPT[idx], return_inverse=True)[1]

# ============================================================ 1. grouping the test pool
def average_linkage_groups(D, thresh):
    """Cluster ids (1-based) from a square distance matrix, average linkage, cut at thresh."""
    if len(D) < 2:
        return np.ones(len(D), dtype=int)
    D = np.asarray(D, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')


def content_cosine_distance(texts, max_features=None):
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=max_features, dtype=np.float32).fit_transform(texts)
    return np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)


def mixed_groups(texts, E, thresh=GROUP_THRESH):
    """Average of the content-word TF-IDF cosine distance and the cosine distance of mean-centred, L2-normalised embeddings E."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    Z = np.asarray(E, dtype=np.float64)
    Z = Z - Z.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    D = 0.5 * (content_cosine_distance(texts) + np.clip(1.0 - Z @ Z.T, 0.0, 2.0))
    return average_linkage_groups(D, thresh)


def normalised_content_groups(texts, thresh=STYLE_GROUP_THRESH):
    """The stylegram's grouping: content distance divided by its median off-diagonal value, cut at thresh."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    D = content_cosine_distance(texts, max_features=100_000)
    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-6)
    return average_linkage_groups(D / med, thresh)


def eligible(groups, lo=GROUP_MIN, hi=GROUP_MAX):
    size = np.bincount(groups)[groups]
    return (size >= lo) & (size <= hi)


def describe_groups(name, groups):
    size = np.bincount(groups)[groups]
    print(f'{name}: {len(groups)} rows | singletons {int((size == 1).sum())}  2-3: {int(((size >= 2) & (size <= 3)).sum())}'
          f'  4-18: {int(((size >= 4) & (size <= 18)).sum())}  >18: {int((size > 18).sum())}', flush=True)


G_TE, GROUP_ROWS = None, None   # set after the GLM pass (section 4)

# ============================================================ 2. style n-grams + extra passages + gates
FUNCTION_WORDS = frozenset("a an the and or but if while although because as so than that which who whom whose this these those it its "
                           "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with "
                           "without for of by over under between through into about is are was were be been being have has had do does did "
                           "can could will would shall should may might must not no nor all any some each every both either neither other "
                           "another such only also just very too more most less least here there when where why how what whether then now "
                           "still yet even much many few little".split())
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def shape_view(text):
    return re.sub(r"[^\W\d_]+", lambda m: 'X' if m[0].isupper() else 'x', text)


def function_view(text):
    return ' '.join(t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else 'CONTENT' for t in TOKEN_RE.findall(text))


class StyleGrams:
    """Four stacked TF-IDF views: char 3-5, word 1-3, shape (char 2-6 of the x/X text), function-word (word 1-3 of the masked text)."""
    VIEWS = [('char', str, (3, 5), 250_000, 1.0), ('word', str, (1, 3), 180_000, 0.7),
             ('char', shape_view, (2, 6), 60_000, 0.5), ('word', function_view, (1, 3), 50_000, 0.5)]

    def fit(self, texts):
        self.vecs = []
        for analyzer, view, ngr, maxf, w in self.VIEWS:
            opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
            if analyzer == 'word':
                opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
            self.vecs.append((TfidfVectorizer(**opts).fit([view(t) for t in texts]), view, w))
        return self

    def transform(self, texts):
        return hstack([v.transform([view(t) for t in texts]) * w for v, view, w in self.vecs], format='csr')


def sparse_group_centre(X, groups, lo=GROUP_MIN, hi=GROUP_MAX):
    """Subtract each group's mean row (sparse); groups outside [lo, hi] are left as they are."""
    X = csr_matrix(X)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep):
        return X.copy()
    col = np.full(len(cnt), -1); col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))
    means = diags(1.0 / cnt[keep]) @ (M.T @ X)
    return (X - M @ means).tocsr()


def svc_probs(svc, X):
    return full_prob(softmax_rows(svc.decision_function(X)), svc.classes_)


# ---- public in-style passages (prompt-bank protocol), one per (prompt, label); prompts answered by >= EXTRA_MIN_LABELS labels
def load_extra_passages():
    sources = [('**/ntl_extra_passages_claude.csv', 'model', None), ('**/gen_passages.csv', 'model', None),
               ('**/ntl_generations.csv', 'label', 'finish_reason'), ('**/ntl_proxy_generations.csv', 'label', None),
               ('**/ntl_generated*.csv', 'model', 'finish'), ('**/passages_clean.csv', 'model', None)]
    parts = []
    for pattern, label_col, finish_col in sources:
        for f in sorted(glob.glob('/kaggle/input/' + pattern, recursive=True)):
            d = pd.read_csv(f)
            if finish_col and finish_col in d:
                d = d[d[finish_col].eq('stop')]
            d = d.rename(columns={label_col: 'model'})[['prompt_id', 'model', 'text']].copy()
            d['prompt_id'] = ('pv_' if f.endswith('passages_clean.csv') else '') + d['prompt_id'].astype(str)   # prvsiyan's prompts are a different bank
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=['prompt_id', 'model', 'text'])
    x = pd.concat(parts, ignore_index=True)
    x = x[x['model'].isin(LABELS)].copy()
    x['text'] = x['text'].astype(str).map(lambda t: re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', t.strip())).strip())
    x = x[x['text'].str.split().str.len().between(30, 300)].drop_duplicates(subset=['model', 'text'])
    x = x.drop_duplicates(subset=['prompt_id', 'model'])
    x = x[x.groupby('prompt_id')['model'].transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)
    return x


EXTRA = load_extra_passages()
EX_T, EX_Y, EX_G = EXTRA['text'].tolist(), EXTRA['model'].map(LAB2I).to_numpy(dtype=int), EXTRA['prompt_id'].to_numpy(dtype=object)
print(f'extra passages: {len(EXTRA)} rows, {EXTRA["prompt_id"].nunique()} prompts', flush=True)


def extra_sample_weights():
    """Per class: original rows keep (1 - EXTRA_SHARE) of the class weight, the class's extra rows share EXTRA_SHARE of it."""
    n_tr, n_ex = np.bincount(y, minlength=K).astype(float), np.bincount(EX_Y, minlength=K).astype(float)
    w_tr = np.where(n_ex[y] > 0, 1.0 - EXTRA_SHARE, 1.0)
    w_ex = EXTRA_SHARE * n_tr[EX_Y] / np.maximum(n_ex[EX_Y], 1.0)
    return np.concatenate([w_tr, w_ex])


# Added raw responses are private to this Stylegram member.
_volume_arm = 'four240'
_volume_rows = 240
_volume_sha = '694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01'
_volume_shared_sha = 'b98c246578afbc38b9bf3c0be1e666a41cc15d051c64d03e875ab0c85da1bd3b'
_volume_b64 = (
    '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'
)


def _volume_load():
    import base64, hashlib, io, zlib
    raw = zlib.decompress(base64.b64decode(_volume_b64, validate=True))
    assert hashlib.sha256(raw).hexdigest() == _volume_sha
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    assert len(frame) == _volume_rows and frame.row_id.is_unique and frame.job_key.is_unique
    assert frame.split.eq('train').all()
    assert frame.text.map(lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest()).tolist() == frame.text_sha256.tolist()
    assert frame.text_sha256.is_unique and frame.text.str.len().gt(0).all()
    classes = {'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'}
    assert frame.model.value_counts().to_dict() == dict.fromkeys(classes, _volume_rows // 4)
    assert frame.prompt_id.nunique() == _volume_rows // 4
    assert frame.groupby('prompt_id').size().eq(4).all() and frame.groupby('prompt_id').model.nunique().eq(4).all()
    return frame


def _volume_check_shared(frame):
    import hashlib
    canonical = frame[['prompt_id', 'model', 'text']].to_csv(index=False, lineterminator='\n').encode('utf-8')
    assert len(frame) == 10743 and frame.prompt_id.nunique() == 738
    assert hashlib.sha256(canonical).hexdigest() == _volume_shared_sha


_volume_data = _volume_load()
_volume_check_shared(EXTRA)
_volume_texts = _volume_data.text.tolist()
_volume_y = _volume_data.model.map(LAB2I).to_numpy(dtype=int)
_volume_groups = _volume_data.prompt_id.to_numpy(dtype=object)
_volume_base_weights = extra_sample_weights()
_volume_ntrain = np.bincount(y, minlength=K)
_volume_npublic = np.bincount(EX_Y, minlength=K)
assert (_volume_npublic[_volume_y] > 0).all()
_volume_weights = np.r_[_volume_base_weights, EXTRA_SHARE * _volume_ntrain[_volume_y] / _volume_npublic[_volume_y]]
_volume_target = np.concatenate([y, EX_Y, _volume_y])

grams = StyleGrams().fit(TR + EX_T + _volume_texts)
X_tr, X_ex, X_te = grams.transform(TR), grams.transform(EX_T), grams.transform(TE)
_volume_x = grams.transform(_volume_texts)


def fit_svc(X, labels, C=0.5, **kw):
    return LinearSVC(C=C, random_state=42, max_iter=4000).fit(X, labels, **kw)


svc_plain = fit_svc(vstack([X_tr, X_ex, _volume_x]).tocsr(), _volume_target, sample_weight=_volume_weights)
P_style = svc_probs(svc_plain, X_te)
X_tr_c = sparse_group_centre(X_tr, PROMPT)
svc_centred = fit_svc(vstack([X_tr_c, sparse_group_centre(X_ex, EX_G, GROUP_MIN, 10 ** 6),
                              sparse_group_centre(_volume_x, _volume_groups, GROUP_MIN, 10 ** 6)]).tocsr(),
                      _volume_target, sample_weight=_volume_weights)
G_style = normalised_content_groups(TE)
rows_style = eligible(G_style)
describe_groups('stylegram groups', G_style)
if rows_style.any():
    P_style[rows_style] = svc_probs(svc_centred, sparse_group_centre(X_te[rows_style], G_style[rows_style]))
if TINY:   # exercise the centred path on training rows grouped by their true prompt (output discarded)
    st_idx, st_g = self_test_rows()
    acc = np.mean(svc_centred.classes_[svc_centred.decision_function(sparse_group_centre(X_tr[st_idx], st_g)).argmax(1)] == y[st_idx])
    print(f'  centred stylegram self-test on {len(st_idx)} training rows (seen in training): acc {acc:.2f}', flush=True)
add_member('stylegram', P_style)
# Reached only after both fits, routing, and the author's add_member call succeeded.
assert np.array_equal(svc_plain.classes_, np.arange(K)) and np.array_equal(svc_centred.classes_, np.arange(K))
assert P_style.shape == (len(TE), K) and np.isfinite(P_style).all() and np.allclose(P_style.sum(1), 1.)
_volume_receipt = {
    'status': 'MEMBER_ADDED', 'arm': _volume_arm, 'run_nonce': os.environ['NTL_VOLUME_RUN_NONCE'],
    'rows': len(_volume_data), 'groups': _volume_data.prompt_id.nunique(), 'source_csv_sha256': _volume_sha,
    'shared_rows': len(EX_T), 'shared_groups': EXTRA.prompt_id.nunique(), 'shared_sha256': _volume_shared_sha,
    'test_rows': len(TE), 'official_rows': len(TR), 'fit_count': 2,
    'output_classes': K, 'member_weight': W['stylegram'], 'centred_rows': int(rows_style.sum()),
    'full_training_rows': len(_volume_target), 'plain_n_iter': int(svc_plain.n_iter_), 'centred_n_iter': int(svc_centred.n_iter_),
    'per_added_class': {
        label: {'added_rows': int(np.sum(_volume_y == LAB2I[label])), 'official_rows': int(_volume_ntrain[LAB2I[label]]),
                'original_public_rows': int(_volume_npublic[LAB2I[label]]),
                'added_per_row_weight': float(EXTRA_SHARE * _volume_ntrain[LAB2I[label]] / _volume_npublic[LAB2I[label]])}
        for label in sorted(set(_volume_data.model))
    },
}
pathlib.Path('/kaggle/working/stylegram_volume_receipt.json').write_text(json.dumps(_volume_receipt, indent=2), encoding='utf-8')
del _volume_x
del X_tr_c, svc_plain, svc_centred, P_style


# ---- gates: style SVCs that only re-rank inside one family after the blend
def family_gate(texts, labels, classes, C, sample_weight=None, balanced=False):
    g = StyleGrams().fit(texts)
    svc = LinearSVC(C=C, class_weight='balanced' if balanced else None, random_state=42, max_iter=4000).fit(
        g.transform(texts), labels, sample_weight=sample_weight)
    assert list(svc.classes_) == list(classes)
    d = svc.decision_function(g.transform(TE))
    return np.column_stack([-d, d]) if len(classes) == 2 else d


GATES = {}
api_claude = sorted(glob.glob('/kaggle/input/**/ntl_extra_passages_claude.csv', recursive=True))
if api_claude:
    api = pd.read_csv(api_claude[0])
    m = np.isin(y, FAMILY_IDX['claude'])
    GATES['claude'] = family_gate([t for t, k in zip(TR, m) if k] + api['text'].astype(str).tolist(),
                                  np.concatenate([y[m], api['model'].map(LAB2I).to_numpy(dtype=int)]), FAMILY_IDX['claude'], 0.5)
    print(f'claude gate: {int(m.sum())} official + {len(api)} API rows', flush=True)
    del api
else:
    print('Claude API passages not attached: Claude gate off', flush=True)
for fam, C in [('gpt', 1.0), ('gemini', 1.0), ('glm', 0.5)]:
    m, me = np.isin(y, FAMILY_IDX[fam]), np.isin(EX_Y, FAMILY_IDX[fam])
    GATES[fam] = family_gate([t for t, k in zip(TR, m) if k] + [t for t, k in zip(EX_T, me) if k], np.concatenate([y[m], EX_Y[me]]),
                             FAMILY_IDX[fam], C, sample_weight=np.concatenate([np.ones(int(m.sum())), np.full(int(me.sum()), 0.25)]), balanced=True)
    print(f'{fam} gate: {int(m.sum())} official + {int(me.sum())} extra rows', flush=True)
del grams, X_tr, X_ex, X_te

# ============================================================ 3. stylometric features + LightGBM
PUNCT = list(',.;:!?()[]"\'-') + ['—', '–', '‘', '’', '“', '”', '…', '*', '/', '%']
STOPS = sorted(ENGLISH_STOP_WORDS)
SENT_RE, WORD_RE = re.compile(r'(?<=[.!?])\s+'), re.compile(r"[A-Za-z’']+")


def stylometry(t):
    words = WORD_RE.findall(t)
    low = [w.lower() for w in words]
    nw, nc = max(len(words), 1), max(len(t), 1)
    sents = [s for s in SENT_RE.split(t.strip()) if s]
    sl = np.array([len(WORD_RE.findall(s)) for s in sents] or [0], dtype=float)
    wl = np.array([len(w) for w in words] or [0], dtype=float)
    f = [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(), (wl >= 9).mean(), len(set(low)) / nw,
         sum(c.isupper() for c in t) / nc, sum(c.isdigit() for c in t) / nc, sum(ord(c) > 127 for c in t) / nc,
         t.count(', ') / nw, t.count(' and ') / nw, t.count(' but ') / nw,
         sum(1 for s in sents if s[:1].isupper() and s.split()[0].lower() in ('this', 'these', 'it', 'they')) / max(len(sents), 1),
         sum(w.endswith('ly') for w in words) / nw, sum(w.endswith('ing') for w in words) / nw, sum(w.endswith('tion') for w in words) / nw]
    f += [t.count(q) / nw for q in PUNCT]
    counts = pd.Series(low).value_counts() if low else pd.Series(dtype=float)
    f += [counts.get(s, 0) / nw for s in STOPS]
    return f


F_tr, F_te = (np.array([stylometry(t) for t in txt], dtype=np.float32) for txt in (TR, TE))
try:
    import lightgbm as lgb
    booster = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4).fit(F_tr, y)
    add_member('stylometry_lgb', full_prob(booster.predict_proba(F_te), booster.classes_))
    del booster
except ImportError:
    assert SMOKE, 'lightgbm missing'
    print('smoke: lightgbm unavailable, stylometry member skipped', flush=True)
del F_tr, F_te

# ============================================================ 4b. shared-prompt Gaussian group head
# Verbatim core of our research tool tools/group_gaussian.py (sha256 8a42427b77a37f56...), numpy only.
# Model per dense view, in a reduced space z (standardise -> within-prompt-residual PCA, fitted on training rows):
#   z_i = mu_{y_i} + p_g + e_i,  p_g ~ N(0, S_p) shared by every passage of prompt group g,  e_i ~ N(0, S_e).
# A joint diagonalisation makes the dimensions independent; for a detected group of n passages the exact joint
# log-likelihood of a label tuple is sum_i a_i(c_i) + sum_{i<j} b_ij(c_i, c_j). Groups of 2-3 are decoded exactly over
# all label tuples with the at-most-once constraint; larger groups by a cavity EM with the finite-population-corrected
# prompt-effect estimate lam / (1 + (n-1) lam); singletons integrate the prompt effect out (never used here).
import itertools
K = 18
DEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "n_exact": 3, "inj": True,
           "iters": 5, "repel": True, "sp_scale": 1.0}

# =================================================================== primitives ====
def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims: return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())

def log_softmax(L): return L - _lse(L, axis=1, keepdims=True)

def _lw_shrinkage(X):
    """Ledoit-Wolf shrinkage coefficient for rows X that are already centred (sklearn formula, assume_centered)."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]; return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, iters=4):
    """additive fit Z = mu[y] + P[prompt] + E by alternating means (balanced design -> converges in 1-2 rounds)."""
    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y == c].mean(0)
        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]
    return mu, P, cnt_p

def fit(X, y, prompts, cfg=None):
    """fit the shared-prompt Gaussian on training rows. X (N, D) raw features, y (N,) labels in 0..17, prompts (N,) group ids.
    Returns a params dict (numpy arrays + floats); temperatures default to 1 (set by the caller's cross-fit)."""
    c = dict(DEFAULT); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)
    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx
    if c["red"] == "none":
        V = np.eye(D)
    else:
        if c["red"] == "wpca":
            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]; R = Xs - Pm[pinv]
        elif c["red"] == "pca": R = Xs
        else: raise ValueError(c["red"])
        d = int(min(c["d"], N - G - 1, D))
        # economy SVD through the (N x N) Gram matrix (N << D for the real views)
        if N < D:
            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]; w, U = np.clip(w[o], 1e-12, None), U[:, o]
            V = (R.T @ U) / np.sqrt(w)
        else:
            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T
    Z = Xs @ V; d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G)
    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"]); Se = _shrunk(Se_raw, ae)
    Pc = P - P.mean(0); Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se)); Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]
    # joint diagonalisation: A Se A' = I, A Sp A' = diag(lam)
    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T; M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,
            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4), "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()}, "cfg": c}

def transform(X, prm): return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]

_MASK = {}
def _inj_mask(n, C):
    if (n, C) not in _MASK:
        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij"); bad = np.zeros([C] * n, bool)
        for i, j in itertools.combinations(range(n), 2): bad |= grids[i] == grids[j]
        _MASK[(n, C)] = bad
    return _MASK[(n, C)]

def singleton_ll(U, m, lam):
    """(n, C) log N(u; m_c, I + diag lam) up to a constant (p integrated out)."""
    R = U[:, None, :] - m[None]; return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def joint_terms(U, m, lam):
    """a (n, C) and B (n, n, C, C) of the exact joint log-likelihood of one group (constant in c dropped)."""
    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]
    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1); G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)
    return a, B

def joint_table(a, B):
    """full tuple table Q[c_1, ..., c_n] = sum_i a_i + sum_{i<j} B_ij."""
    n, C = a.shape; Q = np.zeros([C] * n)
    for i in range(n):
        sh = [1] * n; sh[i] = C; Q = Q + a[i].reshape(sh)
    for i, j in itertools.combinations(range(n), 2):
        sh = [1] * n; sh[i] = C; sh[j] = C; Q = Q + B[i, j].reshape(sh)
    return Q

def exact_marginals(Q, T=1.0, inj=True):
    """per-passage log marginal posteriors (n, C) from a tuple table Q (unnormalised log joint), tempered by 1/T."""
    n = Q.ndim; C = Q.shape[0]; Qt = Q / T
    if inj and n >= 2: Qt = np.where(_inj_mask(n, C), -np.inf, Qt)
    lz = _lse(Qt); out = np.empty((n, C))
    for i in range(n):
        ax = tuple(k for k in range(n) if k != i); out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz
    return out

def em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):
    """cavity EM for one group (any n >= 2): returns (n, C) log q."""
    n = len(U); v = lam / (1 + (n - 1) * lam)
    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))
    q = np.exp(log_softmax(L))
    for _ in range(iters):
        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v                  # cavity prompt-effect means (n, d)
        R = U[:, None, :] - Pm[:, None, :] - m[None]; L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T
        if repel:   # mean-field at-most-once: discourage classes the other members already hold
            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(L))
    return log_softmax(L)

LAM_DIFF = 1e6
def lam_g(prm, cfg=None):
    """lam used for groups: the fitted one, or ~infinity in the paired-difference ablation (cfg 'diff'): then the joint likelihood
    only sees within-group differences (x_i - x_j around mu_a - mu_b with covariance 2 S_e; triples: centred observations vs
    centred prototypes) and the prompt term cancels exactly."""
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {})
    return np.full_like(prm["lam"], LAM_DIFF) if c.get("diff") else prm["lam"]

def group_logpost(U, prm, T1=None, Tg=None, cfg=None):
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U); lg = lam_g(prm, cfg)
    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg
    if n == 1: return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)
    if n <= c["n_exact"]:
        a, B = joint_terms(U, prm["m"], lg); return exact_marginals(joint_table(a, B), Tg, c["inj"])
    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])

def scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):
    """(n, 18) log-probabilities for every pool row; lab = detected group labels of the pool rows (singletons allowed)."""
    U = transform(X_pool, prm) if U is None else U; lab = np.asarray(lab); n = len(U); out = np.empty((n, prm["m"].shape[0]))
    sizes = np.bincount(lab); s1 = sizes[lab] == 1
    T1 = prm["T1"] if T1 is None else T1
    if s1.any(): out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)
    return out

# ============================================================ 4. frozen models: plain heads, centred twins, Gaussian group heads
if not SMOKE:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
CACHED_PARTS = {}   # key -> {part name: (train features, test features)}


def find_checkpoint(model_type, hidden=None, dir_name=None):
    """A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name."""
    pattern = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'
    for cfg in sorted(glob.glob(pattern, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden)):
            return os.path.dirname(cfg)
    return None


def load_truncated_decoder(path, keep, dtype):
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = keep
    if getattr(cfg, 'layer_types', None):
        cfg.layer_types = cfg.layer_types[:keep]
    kw = dict(config=cfg, dtype=dtype)
    if DEV == 'cuda':
        kw['device_map'] = 'auto'
    return AutoModel.from_pretrained(path, **kw).eval()


def style_token_mask(tok):
    """Vocabulary mask of 'style' tokens: the decoded piece is an English stop word or pure punctuation."""
    pieces = [tok.decode([i]) for i in range(len(tok))]
    return torch.tensor([(p.strip().lower() in ENGLISH_STOP_WORDS) or bool(re.fullmatch(r'\s*[^\w\s]+\s*', p)) for p in pieces])


@torch.no_grad()
def pooled_states(model, tok, texts, parts, decoder, bs, max_len=512):
    """{'L{layer}_{pool}': (n, hidden)} masked mean pools of hidden_states[layer] (all tokens, or style tokens only)."""
    H = model.config.hidden_size
    out = {f'L{L}_{pool}': np.zeros((len(texts), H), dtype=np.float32) for L, pool in parts}
    smask = style_token_mask(tok) if any(pool == 'style' for _, pool in parts) else None
    dev = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    retried = 0
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        if decoder:
            batch = [(tok.bos_token or '') + texts[i] for i in idx]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len, add_special_tokens=False).to(dev)
        else:
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if decoder and not all(torch.isfinite(hs[L]).all() for L, _ in parts):   # fp16 overflow: redo the batch in bf16
            retried += 1
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
        m_all = enc['attention_mask'].float()
        for L, pool in parts:
            m = m_all * smask.to(m_all.device)[enc['input_ids']].float() if pool == 'style' else m_all
            m = m.unsqueeze(-1).to(hs[L].device)
            out[f'L{L}_{pool}'][idx] = ((hs[L].float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        del hs, enc
    if retried:
        print(f'  {retried} batches redone in bf16', flush=True)
    return {k: np.nan_to_num(v) for k, v in out.items()}


def pseudo_parts(parts, dim=48):
    """SMOKE only: deterministic stand-ins for the model passes (TF-IDF SVD of the texts), same downstream code."""
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(TR + TE)
    out = {}
    for j, (L, pool) in enumerate(parts):
        svd = TruncatedSVD(dim, random_state=L + 7 * j).fit(tf.transform(TR))
        out[f'L{L}_{pool}'] = (svd.transform(tf.transform(TR)).astype(np.float32), svd.transform(tf.transform(TE)).astype(np.float32))
    return out


def extract_parts(key, parts, loader):
    """Train and test pools for every requested part; loader() returns (model, tok, decoder flag, batch size)."""
    if SMOKE:
        return pseudo_parts(parts)
    model, tok, decoder, bs = loader()
    tr, te = pooled_states(model, tok, TR, parts, decoder, bs), pooled_states(model, tok, TE, parts, decoder, bs)
    del model, tok
    gc.collect()
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return {k: (tr[k], te[k]) for k in tr}


def decoder_loader(model_type, hidden, keep):
    def load():
        path = find_checkpoint(model_type, hidden)
        assert path, f'no {model_type} checkpoint with hidden size {hidden} under /kaggle/input'
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = 'right'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return load_truncated_decoder(path, keep, torch.float16 if DEV == 'cuda' else torch.float32), tok, True, 8
    return load


def encoder_loader(dir_name):
    def load():
        path = find_checkpoint(None, dir_name=dir_name)
        assert path, f'encoder directory {dir_name} not found under /kaggle/input'
        return AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval(), AutoTokenizer.from_pretrained(path), False, 16
    return load


def concat_parts(cache, parts):
    return (np.concatenate([cache[f'L{L}_{p}'][0] for L, p in parts], 1), np.concatenate([cache[f'L{L}_{p}'][1] for L, p in parts], 1))


def plain_and_twin(key, Etr, Ete, C):
    """Plain head (test-side standardisation) for the blend; centred twin evidence on grouped rows."""
    if W[key] > 0:
        sc = StandardScaler().fit(Etr)
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Etr), y)
        Xte = (Ete - Ete.mean(0)) / (Ete.std(0) + 1e-6) if len(Ete) >= 2 else sc.transform(Ete)
        add_member(key, full_prob(clf.predict_proba(Xte), clf.classes_))
    else:
        print(f'  {key}: plain head has weight 0 in the 0.563 blend, not fitted', flush=True)
    if key in TWIN_W:
        csc = StandardScaler().fit(group_centre_dense(Etr, PROMPT))
        twin = LogisticRegression(C=C, max_iter=5000).fit(csc.transform(group_centre_dense(Etr, PROMPT)), y)
        if GROUP_ROWS.any():
            pc = twin.predict_proba(csc.transform(group_centre_dense(Ete[GROUP_ROWS], G_TE[GROUP_ROWS])))
            add_member(key + '_twin', full_prob(pc, twin.classes_), rows=GROUP_ROWS, weight=TWIN_W[key])
        if TINY:
            st_idx, st_g = self_test_rows()
            ps = twin.predict_proba(csc.transform(group_centre_dense(Etr[st_idx], st_g)))
            print(f'  {key} twin self-test on {len(st_idx)} training rows (seen in training): acc {np.mean(twin.classes_[ps.argmax(1)] == y[st_idx]):.2f}', flush=True)


def gaussian_group_head(name, spec, Xtr, Xte):
    """Our addition: fit the shared-prompt Gaussian on the training prompts, score the grouped test rows jointly within their group."""
    prm = fit(Xtr, y, PROMPT, dict(GAUSS_SETTING, d=spec['d']))
    prm['T1'], prm['Tg'] = spec['T1'], spec['Tg']
    print(f'  {name}: d {prm["info"]["d"]}, shrinkage e/p {prm["info"]["a_e"]}/{prm["info"]["a_p"]}, lam pct {prm["info"]["lam_pct"]}', flush=True)
    if GROUP_ROWS.any():
        LP = scores(Xte[GROUP_ROWS], np.unique(G_TE[GROUP_ROWS], return_inverse=True)[1], prm)
        add_member(name, np.exp(LP), rows=GROUP_ROWS, weight=spec['w'])
    if TINY:
        st_idx, st_g = self_test_rows()
        LP = scores(Xtr[st_idx], st_g, prm)
        print(f'  {name} self-test on {len(st_idx)} training rows (seen in training): acc {np.mean(LP.argmax(1) == y[st_idx]):.2f}', flush=True)


for key, model_type, hidden, parts, extra_parts, C in LM_HEADS:
    print(f'[{key}] {model_type} hidden {hidden} parts {parts + extra_parts}', flush=True)
    cache = extract_parts(key, parts + extra_parts, decoder_loader(model_type, hidden, max(L for L, _ in parts + extra_parts) + 1))
    if G_TE is None:   # the GLM pass comes first: its layer-24 pool defines the test groups
        G_TE = mixed_groups(TE, cache[f'L{GROUP_LAYER}_mean'][1])
        GROUP_ROWS = eligible(G_TE)
        describe_groups('test groups (content + GLM L24)', G_TE)
        if SMOKE:
            same_g, same_p = G_TE[:, None] == G_TE[None, :], test['prompt_id'].to_numpy(dtype=object)[:, None] == test['prompt_id'].to_numpy(dtype=object)[None, :]
            off = ~np.eye(len(TE), dtype=bool)
            print(f'  smoke grouping: pair precision {(same_g & same_p & off).sum() / max((same_g & off).sum(), 1):.2f} '
                  f'recall {(same_g & same_p & off).sum() / max((same_p & off).sum(), 1):.2f}', flush=True)
    CACHED_PARTS[key] = cache
    plain_and_twin(key, *concat_parts(cache, parts), C)
    for name, spec in GAUSS_HEADS.items():
        if spec['src'] == key:
            gaussian_group_head(name, spec, *cache[spec['part']])
    del cache
    CACHED_PARTS.pop(key)
    gc.collect()

for key, dir_name, parts, C in ENC_HEADS:
    print(f'[{key}] {dir_name} parts {parts}', flush=True)
    cache = extract_parts(key, parts, encoder_loader(dir_name))
    plain_and_twin(key, *concat_parts(cache, parts), C)
    for name, spec in GAUSS_HEADS.items():
        if spec['src'] == key:
            gaussian_group_head(name, spec, *cache[spec['part']])
    del cache
    gc.collect()

# ============================================================ 5. LoRA classifiers (inference only)
@torch.no_grad()
def score_lora_checkpoints(trunk_spec, pool_layers, checkpoints):
    """One truncated trunk, adapters swapped in turn. Plain readout -> blend member; centred readout -> twin on grouped rows;
    within-Claude readout -> Claude gate."""
    import importlib.util as iu
    _find = iu.find_spec   # Kaggle's torchao build is rejected by peft; hide it (unused here)
    iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _find(name, *a, **k)
    from peft import PeftModel
    import peft.import_utils as pi
    pi.is_torchao_available = lambda *a, **k: False
    from safetensors.torch import load_file
    model_type, H, keep = trunk_spec
    base = find_checkpoint(model_type, H)
    assert base, f'no {model_type} base checkpoint under /kaggle/input'
    cfg = AutoConfig.from_pretrained(base)
    cfg.num_hidden_layers, cfg.use_cache = keep, False
    if getattr(cfg, 'layer_types', None) is not None:
        cfg.layer_types = cfg.layer_types[:keep]
    trunk = AutoModel.from_pretrained(base, config=cfg, dtype=torch.float16, device_map='auto').eval()
    for key, marker, norms_file, anchor_file in checkpoints:
        hits = sorted(glob.glob('/kaggle/input/' + marker, recursive=True))
        if not hits:   # upstream training kernel no longer public: drop this member, keep the rest of the stack
            print(f'  SKIP {key}: checkpoint not attached ({marker})', flush=True); continue
        ck = pathlib.Path(os.path.dirname(hits[0]))
        print(f'[{key}] {ck}', flush=True)
        tok = AutoTokenizer.from_pretrained(ck / 'tokenizer')
        tok.padding_side = 'right'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lm = PeftModel.from_pretrained(trunk, ck / 'adapter').eval()
        state = load_file(str(ck / norms_file))
        norms = torch.nn.ModuleList([torch.nn.LayerNorm(H) for _ in pool_layers])
        norms.load_state_dict({k[len('norms.'):]: v for k, v in state.items() if k.startswith('norms.')})
        ro = np.load(ck / 'readouts.npz')
        assert list(ro['classes']) == LABELS and list(ro['pool_layers']) == pool_layers
        dev = lm.get_input_embeddings().weight.device
        norms = norms.to(dev).float()

        def token_ids(texts):
            return [ids + [tok.eos_token_id] for ids in tok(list(texts), add_special_tokens=False, truncation=True, max_length=LORA_MAX_LEN)['input_ids']]   # these checkpoints' anchors match the tokenizer's own eos

        def features(texts, bs):
            ids = token_ids(texts)
            Z = np.zeros((len(ids), H * len(pool_layers)), dtype=np.float32)
            order = np.argsort([len(t) for t in ids], kind='stable')
            for b0 in range(0, len(order), bs):
                idx = order[b0:b0 + bs]
                batch = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(dev)
                with torch.autocast('cuda', dtype=torch.float16):
                    hs = lm(**batch, output_hidden_states=True, use_cache=False).hidden_states
                m = batch['attention_mask'].unsqueeze(-1).float()
                m[:, 0] = 0   # the first token is masked out of the mean, as in training
                Z[idx] = torch.cat([nm((hs[L].float() * m.to(hs[L].device)).sum(1).to(dev) / m.sum(1).to(dev))
                                    for L, nm in zip(pool_layers, norms)], -1).cpu().numpy()
                del hs, batch
            return Z

        if (ck / anchor_file).exists():   # the checkpoint's own features for training rows 0..2: our forward pass must reproduce them
            saved = np.load(ck / anchor_file)['features'].astype(np.float32)
            ours = features(TR[:3], 3)
            rel = float(np.abs(ours - saved).mean() / (np.abs(saved).mean() + 1e-9))
            print(f'  parity with the checkpoint anchors: max abs {float(np.abs(ours - saved).max()):.4f}, mean rel {rel:.5f}', flush=True)
            assert rel < 0.02, f'{key}: our features do not match the checkpoint'
        Z = features(TE, LORA_BS)
        add_member(key, softmax_rows(Z @ ro['plain_coef'].T + ro['plain_intercept']))
        if 'claude_coef' in ro.files:
            lc = Z @ ro['claude_coef'].T + ro['claude_intercept']
            GATES['claude_lora'] = lc - np.log(np.exp(lc - lc.max(1, keepdims=True)).sum(1, keepdims=True)) - lc.max(1, keepdims=True)
        if key in LORA_TWIN_W and GROUP_ROWS.any():
            Zc = group_centre_dense(Z[GROUP_ROWS], G_TE[GROUP_ROWS])
            add_member(key + '_twin', softmax_rows(Zc @ ro['center_coef'].T + ro['center_intercept']), rows=GROUP_ROWS, weight=LORA_TWIN_W[key])
        trunk = lm.unload()
        del lm, Z
        gc.collect()
        torch.cuda.empty_cache()
    del trunk
    gc.collect()
    torch.cuda.empty_cache()


if not SMOKE:
    for trunk_spec, pool_layers, checkpoints in LORA_RUNS:
        score_lora_checkpoints(trunk_spec, pool_layers, checkpoints)

# ============================================================ 6. token-attention heads + UltraBERT
class TokenAttention(torch.nn.Module if torch is not None else object):   # the module of the public training kernels (lucasboesen/name-that-llm-train-tokattn, -tokattn2)
    def __init__(self, k, h=4096):
        super().__init__()
        self.drop, self.proj, self.score, self.out = torch.nn.Dropout(0.2), torch.nn.Linear(h, 64), torch.nn.Linear(64, 1), torch.nn.Linear(128, k)

    def forward(self, x, m):
        h = torch.tanh(self.proj(self.drop(x)))
        a = torch.softmax(self.score(h).squeeze(-1).masked_fill(~m, -1e4), -1).unsqueeze(-1)
        mf = m.unsqueeze(-1).float()
        return self.out(self.drop(torch.cat([(h * mf).sum(1) / mf.sum(1).clamp(min=1), (a * h).sum(1)], -1)))


@torch.no_grad()
def token_attention_probs(texts, head_file, backbone, bs=8):
    hits = sorted(glob.glob('/kaggle/input/**/' + head_file, recursive=True))
    assert hits, f'training-kernel output holding {head_file} not attached'
    d = torch.load(hits[0], map_location='cpu')
    assert list(d['classes']) == LABELS
    decoder = d.get('decoder', True)
    path = find_checkpoint('qwen3_5') if backbone == 'qwen3_5' else find_checkpoint(None, dir_name='deberta-v3-large')
    assert path, f'backbone {backbone} not attached'
    cfg = AutoConfig.from_pretrained(path)
    if decoder:
        tc = getattr(cfg, 'text_config', None) or cfg
        tc.num_hidden_layers = d['keep']
        if getattr(tc, 'layer_types', None):
            tc.layer_types = tc.layer_types[:d['keep']]
        lm = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()
    else:
        lm = AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    H = d.get('hidden', 4096)
    net = TokenAttention(K, H)
    net.load_state_dict(d['state_dict'])
    dev = lm.get_input_embeddings().weight.device
    net, mu, sd = net.to(dev).eval(), d['mu'].float().to(dev), d['sd'].float().to(dev)
    ids = tok(list(texts), add_special_tokens=not decoder, truncation=True, max_length=d['max_len'])['input_ids']
    P = np.zeros((len(texts), K))
    order = np.argsort([len(t) for t in ids])
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        batch = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(dev)
        out = lm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        hs = (out.hidden_states if getattr(out, 'hidden_states', None) is not None else out.language_model_outputs.hidden_states)[d['layer']]
        n = batch['attention_mask'].sum(1)
        T = max(int(n.max()) - 1, 1)
        x = torch.zeros(len(idx), T, H, device=dev)
        m = torch.zeros(len(idx), T, dtype=torch.bool, device=dev)
        for r in range(len(idx)):   # first token dropped, states rounded through fp16, as in training
            k = int(n[r]) - 1
            if k > 0:
                x[r, :k] = ((torch.nan_to_num(hs[r, 1:k + 1].to(dev).float()).half().float() - mu) / sd).clamp(-8, 8)
                m[r, :k] = True
        P[idx] = torch.softmax(net(x, m), -1).cpu().numpy()
        del hs, out, batch
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return P


@torch.no_grad()
def ultrabert_probs(texts):
    from transformers import AutoModelForSequenceClassification
    folds = None
    for root in glob.glob('/kaggle/input/**/ultrabert-380m-ntl', recursive=True):
        fs = [f for f in sorted(glob.glob(os.path.join(root, 'fold*'))) if os.path.exists(os.path.join(f, 'config.json'))
              and json.load(open(os.path.join(f, 'config.json'))).get('model_type') == 'ultrabert']
        if len(fs) == 5:
            folds = fs
            break
    assert folds, 'ultrabert-380m-ntl (5 folds) not attached'
    tok = AutoTokenizer.from_pretrained(folds[0], trust_remote_code=True)
    ids = tok(texts, truncation=True, max_length=512)['input_ids']
    order = np.argsort([len(e) for e in ids])
    P = np.zeros((len(texts), K))
    for fp in folds:
        em = AutoModelForSequenceClassification.from_pretrained(fp, trust_remote_code=True, dtype=torch.float16, attn_implementation='sdpa').to(DEV).eval()
        cols = [LAB2I[em.config.id2label[i]] for i in range(len(em.config.id2label))]
        for b0 in range(0, len(order), 32):
            idx = order[b0:b0 + 32]
            b = tok.pad({'input_ids': [ids[j] for j in idx]}, return_tensors='pt').to(DEV)
            P[np.ix_(idx, cols)] += em(**b).logits.float().softmax(-1).cpu().numpy() / len(folds)
        del em
        gc.collect()
        torch.cuda.empty_cache()
    return P


if not SMOKE:
    # Optional community heads are enabled when both the trained head and matching backbone are attached.
    _available_tokattn = []
    for _key, _head_file, _backbone in TOKATTN_HEADS:
        _head_hits = glob.glob('/kaggle/input/**/' + _head_file, recursive=True)
        _base_hit = find_checkpoint('qwen3_5') if _backbone == 'qwen3_5' else find_checkpoint(None, dir_name='deberta-v3-large')
        if _head_hits and _base_hit:
            _available_tokattn.append((_key, _head_file, _backbone))
        else:
            print(f'  OPTIONAL HEAD SKIPPED {_key}: head={bool(_head_hits)} backbone={bool(_base_hit)}', flush=True)
    TOKATTN_HEADS = _available_tokattn
    for key, head_file, backbone in TOKATTN_HEADS:
        if TINY:   # in-sample check on training rows the head was fitted on: must be high, or the checkpoint is not being reproduced
            acc = float(np.mean(token_attention_probs(TR[:40], head_file, backbone).argmax(1) == y[:40]))
            print(f'  {key} in-sample check on 40 training rows: acc {acc:.2f}', flush=True)
            assert acc > 0.5, f'{key} does not reproduce its training fit'
        add_member(key, token_attention_probs(TE, head_file, backbone))
    add_member('ultrabert', ultrabert_probs(TE))


def _np_logsoftmax(z):
    from scipy.special import logsumexp as _lse_np
    z = np.asarray(z, dtype=np.float64)
    return z - _lse_np(z, axis=1, keepdims=True)

def _apply_conditional_family_gate(gate_key, family, lam, scores_are_logprob=False):
    '''Use the already-trained family gate only to rerank siblings, preserving total family mass.
    Confidence-aware interpolation avoids overriding high-confidence cross-family decisions.'''
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
    # Preserve the ensemble's probability mass assigned to this family; only adjust within-family identity.
    from scipy.special import logsumexp as _lse_np
    family_mass = _lse_np(base, axis=1)
    LOGIT[:, idx] = family_mass[:, None] + mixed_lp
    print(f'  family gate applied {gate_key}: mean strength={strength.mean():.4f}, active rows={int((strength > 0).sum())}', flush=True)

# These gates were fitted above but were previously never added to LOGIT. Apply them conservatively.
if not SMOKE:
    _apply_conditional_family_gate('claude', 'claude', GATE_LAM['claude_style'])
    _apply_conditional_family_gate('claude_lora', 'claude', GATE_LAM['claude_lora'], scores_are_logprob=True)
    for _fam in ('gpt', 'gemini', 'glm'):
        _apply_conditional_family_gate(_fam, _fam, GATE_LAM['family'])

np.save('/kaggle/working/comp_logit.npy', LOGIT.astype(np.float32))
json.dump({'labels': LABELS, 'ids': test['id'].astype(str).tolist(), 'seconds': time.time() - T0,
           'family_gates_applied': bool(not SMOKE)}, open('/kaggle/working/comp_meta.json', 'w'))
print(f'COMP STACK DONE {time.time() - T0:.0f}s', flush=True)
