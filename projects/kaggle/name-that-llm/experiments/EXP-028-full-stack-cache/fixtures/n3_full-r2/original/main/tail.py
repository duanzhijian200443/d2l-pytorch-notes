# Harness-selected original math; model paths are excluded.
COMP_P = None
if a.comp_w > 0 and a.comp_path and os.path.exists(a.comp_path):
    try:
        _cm = json.load(open(a.comp_path.replace('comp_logit.npy', 'comp_meta.json')))
        assert _cm['ids'] == test['id'].astype(str).tolist(), 'comp stack row order differs'
        _cl = np.load(a.comp_path)[:, [_cm['labels'].index(l) for l in LABELS]]
        COMP_P = np.exp(_cl - _cl.max(1, keepdims=True))
        COMP_P /= COMP_P.sum(1, keepdims=True)
        assert np.isfinite(COMP_P).all()
        log(f"comp stack loaded ({_cm['seconds']:.0f}s)")
    except Exception as _e:
        log('COMP SKIPPED:', repr(_e))
        COMP_P = None
elif a.comp_w > 0:
    log('COMP SKIPPED: no stack output')
_bundles = [d for d in a.atlas_dir.split(',') if d]
_aw = [float(x) for x in str(a.atlas_w).split(',')] if _bundles else []
if _bundles and len(_aw) == 1:
    _aw = _aw * len(_bundles)
if a.sinkhorn > 0 and len(test) >= 18 or _bundles or COMP_P is not None:
    wsum = sum((abs(v).max() > 0 for v in LP.values()))
    lpn = tot / max(1, wsum) if a.sk_div else tot
    P = np.exp(lpn - lpn.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    if COMP_P is not None:
        P = (1 - a.comp_w) * P + a.comp_w * COMP_P
        P /= P.sum(1, keepdims=True)
    if _bundles and sum(_aw) > 0:
        _s = np.sort(P, 1)
        _margin = _s[:, -1] - _s[:, -2]
        rows = np.where(_margin < a.atlas_margin)[0] if a.atlas_margin > 0 and len(test) >= 50 else np.arange(len(test))
        log(f'atlas on {len(rows)}/{len(test)} rows (top-2 margin < {a.atlas_margin})')
        mixed = P.copy()
        _wleft = 1.0
        for _bd, _w in zip(_bundles, _aw):
            try:
                A_, scored = atlas_scores(_bd, rows, a.atlas_budget)
                completed_rows = rows[scored]
                if len(completed_rows):
                    mixed[completed_rows] += _w * (A_[scored] - P[completed_rows])
                log(f'atlas {_bd} mixed {len(completed_rows)}/{len(rows)} rows at {_w}', gpu_mem())
            except Exception as _e:
                log('ATLAS SKIPPED:', _bd, repr(_e))
        P = mixed / mixed.sum(1, keepdims=True)
    Q = P.copy()
    if not (a.sinkhorn > 0 and len(test) >= 18):
        tot = np.log(P + 1e-12)
    if a.sinkhorn > 0 and len(test) >= 18:
        _rho = float(np.clip(a.sinkhorn_target_mix, 0.0, 1.0))
        _pred_marg = np.clip(P.mean(0), 1e-06, None)
        _pred_marg /= _pred_marg.sum()
        _train_prior = np.bincount(y, minlength=18).astype(np.float64) + 1.0
        _train_prior /= _train_prior.sum()
        _target = (1.0 - _rho) * _pred_marg + _rho * _train_prior
        _target = np.clip(_target, 1e-06, None)
        _target /= _target.sum()
        for _ in range(100):
            Q *= len(Q) * _target[None, :] / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
            Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
        tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)
_v38_before_hard = tot.copy()
if a.post and a.perpassage:
    _nfix = 0
    for _mark, _allowed in GUARDS:
        _forb = np.array([l not in _allowed for l in LABELS])
        for _i, _t in enumerate(test.text.values):
            if _mark in _t and LABELS[int(tot[_i].argmax())] not in _allowed:
                tot[_i, _forb] = -1000000000.0
                _nfix += 1
    _sz = np.bincount(Gte)[Gte]
    _nh = 0
    for _g in np.unique(Gte[(_sz >= 12) & (_sz <= 18)]):
        _ii = np.where(Gte == _g)[0]
        _r, _c = linear_sum_assignment(-tot[_ii])
        _nh += int((tot[_ii].argmax(1)[_r] != _c).sum())
        tot[_ii[_r]] = -1000000000.0
        tot[_ii[_r], _c] = 0.0
    log(f'post: em-dash guard fixed {_nfix}, dense-cluster Hungarian changed {_nh}')
pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)
if a.sim:
    from sklearn.metrics import balanced_accuracy_score as bas
    yt = test_truth.model.map(L2I).values
    for k, v_ in LP.items():
        log(f'  {k:12s} arg {bas(yt, v_.argmax(1)):.4f} hung {bas(yt, hungarian(v_, Gte)):.4f}')
    log(f'ENSEMBLE arg {bas(yt, tot.argmax(1)):.4f} | hung(est clusters) {bas(yt, pred):.4f} | hung(true prompts) {bas(yt, hungarian(tot, test_truth.prompt_id.values)):.4f}')
    np.save(os.path.join(a.cache, f'sim{a.sim}_lp.npy'), tot)
if a.save_lp:
    wtot = sum((abs(float(s_.get('w', 1.0))) for s_ in specs)) + (a.tfidf_w if a.tfidf_w > 0 else 0) + (a.distort_w if a.distort_w > 0 else 0) + (a.svc_w if a.svc_w > 0 else 0)
    from scipy.special import log_softmax as _lsm
    np.save(a.save_lp, _lsm(tot / max(wtot, 1e-06), axis=1).astype(np.float32))
    log('saved', a.save_lp, 'norm', wtot)
sub = pd.DataFrame({'id': test.id.values, 'model': [LABELS[i] for i in pred]})
assert sub.model.isin(LABELS).all() and sub.id.is_unique and (len(sub) == len(test))
sub.to_parquet(a.out, index=False)
log('wrote', a.out, sub.model.value_counts().to_dict())
if not a.sim:
    import pathlib as _V38Path
    import scipy.special as _v38sp
    import json as _v38json
    _out = _V38Path.Path(a.out)
    _marks = [' — ', '—', '–', '’', '‘', '“', '”', '«', '»', '"', ';', ':', '(', ')', '[', ']', '**', '`', '…', '•', '\n', '\n\n', 'e.g.', 'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\t', ' - ', ' = ', '\n- ', '\n1.', '###', '```', '> ']
    _Xtr = np.array([[float(_m in _t) for _m in _marks] for _t in train.text.values], dtype=np.float32)
    _Xte = np.array([[float(_m in _t) for _m in _marks] for _t in test.text.values], dtype=np.float32)
    _freq = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    _hits = np.stack([_Xtr[y == _k].sum(axis=0) for _k in range(len(LABELS))], axis=0).astype(np.float64)
    _global_rate = (_Xtr.sum(axis=0) + 2.0) / (len(_Xtr) + 4.0)
    _kappa = 24.0
    _rate = (_hits + _kappa * _global_rate[None, :]) / (_freq[:, None] + _kappa)
    _rate = np.clip(_rate, 0.01, 0.99)
    _global_rate = np.clip(_global_rate, 0.01, 0.99)
    _log_rate = np.log(_rate)
    _log_not_rate = np.log1p(-_rate)
    _base_log_rate = np.log(_global_rate)
    _base_log_not_rate = np.log1p(-_global_rate)
    _signal = np.sqrt(np.mean((_rate - _global_rate[None, :]) ** 2, axis=0))
    _fw = _signal / max(float(np.median(_signal[_signal > 0])) if np.any(_signal > 0) else 1.0, 1e-06)
    _fw = np.clip(_fw, 0.25, 2.0)
    _evidence = _Xte * _fw[None, :] @ _log_rate.T + (1.0 - _Xte) * _fw[None, :] @ _log_not_rate.T
    _evidence -= (_Xte * _fw[None, :] @ _base_log_rate + (1.0 - _Xte) * _fw[None, :] @ _base_log_not_rate)[:, None]
    _evidence -= _evidence.mean(axis=1, keepdims=True)
    _evidence = np.clip(1.5 * _evidence / max(float(_fw.sum()), 1e-06), -0.3, 0.3)
    _raw = _v38_before_hard.copy()
    _probs = _v38sp.softmax(_raw, axis=1)
    _sorted = np.sort(_probs, axis=1)
    _margin = _sorted[:, -1] - _sorted[:, -2]
    _confidence_gate = np.clip((0.6 - _margin) / 0.6, 0.0, 1.0)
    _word_count = np.array([len(re.findall('\\b\\w+\\b', str(_t))) for _t in test.text.values], dtype=np.float32)
    _length_gate = np.clip(np.sqrt(np.maximum(_word_count, 1.0) / 80.0), 0.4, 1.0)
    _active_markers = _Xte.sum(axis=1)
    _coverage_gate = np.clip(np.sqrt((_active_markers + 1.0) / 4.0), 0.45, 1.0)
    _gate = _confidence_gate * _length_gate * _coverage_gate
    _safe_logits = _raw + _gate[:, None] * _evidence

    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = _v31_sub(_safe_logits)
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()
    with open(_out.with_name('v31_diagnostics.json'), 'w') as _fo:
        _v38json.dump({'method': 'hierarchical_beta_bernoulli_typography_v2', 'fit_scope': 'full official train labels, used only for final submission inference; not an OOF validation claim', 'pre_bayes_changed': int((_raw_sub.model.values != _safe.model.values).sum()), 'trained_rows': int(len(train)), 'tested_rows': int(len(test)), 'marker_count': int(len(_marks)), 'mean_gate': float(_gate.mean()), 'length_gate_mean': float(_length_gate.mean()), 'coverage_gate_mean': float(_coverage_gate.mean()), 'sinkhorn': float(a.sinkhorn), 'sinkhorn_target_mix': float(a.sinkhorn_target_mix)}, _fo, indent=2)
    log('V31 Bayesian typography completed; changed from pre-Bayes:', int((_raw_sub.model.values != _safe.model.values).sum()))
