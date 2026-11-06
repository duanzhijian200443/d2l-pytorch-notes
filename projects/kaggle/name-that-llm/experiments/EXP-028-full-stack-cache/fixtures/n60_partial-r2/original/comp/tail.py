# Harness-selected original math; model paths are excluded.
if not SMOKE:
    _apply_conditional_family_gate('claude', 'claude', GATE_LAM['claude_style'])
    _apply_conditional_family_gate('claude_lora', 'claude', GATE_LAM['claude_lora'], scores_are_logprob=True)
    for _fam in ('gpt', 'gemini', 'glm'):
        _apply_conditional_family_gate(_fam, _fam, GATE_LAM['family'])
np.save('C:\\计算机学习\\深度学习\\projects\\kaggle\\name-that-llm\\experiments\\EXP-028-full-stack-cache\\fixtures\\n60_partial-r2\\original\\comp\\comp_logit.npy', LOGIT.astype(np.float32))
json.dump({'labels': LABELS, 'ids': test['id'].astype(str).tolist(), 'seconds': time.time() - T0, 'family_gates_applied': bool(not SMOKE)}, open('C:\\计算机学习\\深度学习\\projects\\kaggle\\name-that-llm\\experiments\\EXP-028-full-stack-cache\\fixtures\\n60_partial-r2\\original\\comp\\comp_meta.json', 'w'))
print(f'COMP STACK DONE {time.time() - T0:.0f}s', flush=True)
