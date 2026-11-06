import re
import numpy as np
from functools import lru_cache
MARKERS = [',','.', ';', ':', '!', '?','—','–','-','’','‘','“','”','"',"'",'(',')','[',']','{','}','*','`','•','…','/','\\',':','\n','\n\n','\n-','\n1.','\n2.','##','```','**','> ','\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE=re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE=re.compile(r'(?<=[.?!])\s+')
@lru_cache(maxsize=15000)
def row(t):
    t = str(t)
    s = WRE.findall(t); lo=[z.lower() for z in s]; n=max(len(s),1); nt=max(len(t),1); ns=max(len(t.splitlines()),1)
    lens=np.array([len(z) for z in s] or [0]); sent_lens=np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n),np.log1p(nt), nt/n,ns/n,len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)),float(np.max(sent_lens)),float(np.mean(lens>=8)),sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt,sum(ord(c)>127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c=='\n' for c in t)/nt]
    base += [min(t.count(m)/n,3) for m in MARKERS]
    cnt = {}
    for w in lo: cnt[w]=cnt.get(w,0)+1
    base += [cnt.get(w,0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)
def features(texts):
    return np.stack([row(str(t)) for t in texts])


def apply_grouped_style_residual(logits, train_text, test_text, labels, y, groups, max_alpha=0.08, diagnostics_path=None):
    """OOF-estimated, conservative sibling re-ranking. Cross-family probability mass is unchanged."""
    import json
    import numpy as np
    from scipy.special import logsumexp, softmax
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    Xt = features(train_text)
    Xq = features(test_text)
    out = np.asarray(logits, dtype=np.float64).copy()
    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)
    diag = {}
    for family, C in [('claude', 0.003), ('gemini', 0.003), ('gpt', 0.003), ('glm', 0.03)]:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]
        oof = np.empty(len(target), dtype=np.int32)
        for a, b in GroupKFold(n_splits=5).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
            model.fit(X[a], target[a])
            oof[b] = model.predict(X[b])
        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        if reliability <= 0:
            diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance, alpha_mean=0.)
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)
        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        # Only modestly modify disputed sibling identities; preserve total provider mass.
        gate = np.clip((0.65 - margin) / 0.65, 0., 1.)
        strength = (max_alpha * reliability) * gate
        mix = (1. - strength[:, None]) * old_p + strength[:, None] * style_p
        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)
        diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance,
                            alpha_mean=float(strength.mean()), alpha_max=float(strength.max()),
                            changed_within_family=int(np.sum(old_p.argmax(1) != mix.argmax(1))))
    if diagnostics_path:
        with open(diagnostics_path, 'w') as f: json.dump(diag, f, indent=2)
    return out
