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
_volume_arm = 'four120'
_volume_rows = 120
_volume_sha = 'a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d'
_volume_shared_sha = 'b98c246578afbc38b9bf3c0be1e666a41cc15d051c64d03e875ab0c85da1bd3b'
_volume_b64 = (
    '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'
    'qYMYati8G1I+C77+X2L1qDY='
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
