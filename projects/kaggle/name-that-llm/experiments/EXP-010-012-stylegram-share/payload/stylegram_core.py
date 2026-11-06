import re
import numpy as np
from scipy.sparse import csr_matrix, diags, hstack
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
GROUP_MIN, GROUP_MAX, STYLE_GROUP_THRESH = 2, 18, 0.90
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

def normalised_content_groups(texts, thresh=STYLE_GROUP_THRESH):
    """The stylegram's grouping: content distance divided by its median off-diagonal value, cut at thresh."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    D = content_cosine_distance(texts, max_features=100000)
    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-06)
    return average_linkage_groups(D / med, thresh)

def eligible(groups, lo=GROUP_MIN, hi=GROUP_MAX):
    size = np.bincount(groups)[groups]
    return (size >= lo) & (size <= hi)
FUNCTION_WORDS = frozenset('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
TOKEN_RE = re.compile('\\w+|[^\\w\\s]')

def shape_view(text):
    return re.sub('[^\\W\\d_]+', lambda m: 'X' if m[0].isupper() else 'x', text)

def function_view(text):
    return ' '.join((t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else 'CONTENT' for t in TOKEN_RE.findall(text)))

class StyleGrams:
    """Four stacked TF-IDF views: char 3-5, word 1-3, shape (char 2-6 of the x/X text), function-word (word 1-3 of the masked text)."""
    VIEWS = [('char', str, (3, 5), 250000, 1.0), ('word', str, (1, 3), 180000, 0.7), ('char', shape_view, (2, 6), 60000, 0.5), ('word', function_view, (1, 3), 50000, 0.5)]

    def fit(self, texts):
        self.vecs = []
        for analyzer, view, ngr, maxf, w in self.VIEWS:
            opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
            if analyzer == 'word':
                opts['token_pattern'] = '(?u)\\b\\w+\\b|[^\\w\\s]'
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
    col = np.full(len(cnt), -1)
    col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))
    means = diags(1.0 / cnt[keep]) @ (M.T @ X)
    return (X - M @ means).tocsr()
