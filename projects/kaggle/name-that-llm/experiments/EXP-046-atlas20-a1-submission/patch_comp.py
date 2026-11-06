"""Strict byte-anchored A1-only patch; returns bytes and a reviewable diff."""
import difflib
import hashlib

SOURCE_SHA256 = 'b6b6a7b8adbe50580c2d0de5a41d0764ae752fe729e1471872a9197ac956f222'
START = b'grams = StyleGrams().fit(TR + EX_T)'
GATES = b'# ---- gates: style SVCs that only re-rank inside one family after the blend'
CLEANUP = b'del grams, X_tr, X_ex, X_te'


def patch_source(source):
    """Patch exactly one frozen source; do not execute it or write any file."""
    if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Original comp_stack.py SHA256 differs from frozen EXP009 source')
    if source.count(START) != 1 or source.count(GATES) != 1 or source.count(CLEANUP) != 1:
        raise ValueError('Stylegram/gate/cleanup anchors are not unique')
    newline = b'\r\n' if b'\r\n' in source else b'\n'
    begin, end = source.index(START), source.index(GATES)
    cleanup_start = source.index(CLEANUP)
    cleanup_end = cleanup_start + len(CLEANUP)
    if not begin < end < cleanup_start or source[cleanup_end:cleanup_end+len(newline)] != newline:
        raise ValueError('Unexpected Stylegram/gate/cleanup order')
    replacement = newline.join(line.encode() for line in [
        '# EXP046 A1: local Stylegram-only data replacement. Gate EX_T/EX_Y/EX_G remain untouched.',
        'import sys as _a1_sys',
        "_a1_module_dir = pathlib.Path(os.environ['NTL_A1_MODULE_DIR']).resolve()",
        "if _a1_module_dir != pathlib.Path('/kaggle/working/ntl_a1_payload'):",
        "    raise ValueError('A1 modules must come from the frozen working payload directory')",
        '_a1_sys.path.insert(0, str(_a1_module_dir))',
        'import a1_stylegram as _a1_module',
        "if pathlib.Path(_a1_module.__file__).resolve() != _a1_module_dir / 'a1_stylegram.py':",
        "    raise ValueError('Wrong imported A1 module path')",
        "P_style, _a1_metadata = _a1_module.predict(TR, y, PROMPT, TE, os.environ['NTL_A1_EXTRA'], labels=LABELS)",
        "G_style, rows_style = _a1_metadata['groups'], _a1_metadata['centred_mask']",
        "describe_groups('stylegram groups', G_style)",
        "print('A1_STYLEGRAM_RECEIPT ' + json.dumps(_a1_metadata['receipt'], sort_keys=True), flush=True)",
        "add_member('stylegram', P_style)",
        'del P_style, _a1_metadata, _a1_module, _a1_module_dir, _a1_sys', '', '',
    ])
    replacement_cleanup = b'# A1 Stylegram matrices and fitted models are local to a1_stylegram.predict.'
    result = source[:begin] + replacement + source[end:cleanup_start] + replacement_cleanup + source[cleanup_end:]
    family_gate_bytes = source[end:cleanup_start]
    if result[result.index(GATES):result.index(replacement_cleanup)] != family_gate_bytes:
        raise ValueError('Family gate code changed')
    compile(result, 'comp_stack_a1.py', 'exec')
    receipt = {'source_sha256': SOURCE_SHA256, 'patched_sha256': hashlib.sha256(result).hexdigest(),
               'changed_regions': ['Stylegram-only training/prediction block', 'obsolete Stylegram-only cleanup'],
               'family_gate_bytes_equal': True, 'family_gate_sha256': hashlib.sha256(family_gate_bytes).hexdigest(),
               'global_extra_loader_bytes_equal': result[:begin] == source[:begin],
               'suffix_after_cleanup_bytes_equal': result[result.index(replacement_cleanup)+len(replacement_cleanup):] == source[cleanup_end:],
               'unified_diff': ''.join(difflib.unified_diff(source.decode().splitlines(keepends=True), result.decode().splitlines(keepends=True),
                                                         fromfile='original/comp_stack.py', tofile='A1/comp_stack.py'))}
    return result, receipt
