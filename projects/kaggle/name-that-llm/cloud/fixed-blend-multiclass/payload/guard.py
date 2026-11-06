"""Supervise one CPU process; retain evidence when budget or validation fails.

The output directory must be empty and the exclusive lock is retained after
completion. RSS is sampled, not an OS-enforced instantaneous memory limit.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import traceback

import psutil


HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results'))
DYNAMIC_NAMES = {'state.json', 'guard.json', 'training.log', '.run.lock', 'artifacts.json'}


def save(name, data):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def claim_output():
    """Reject an existing run without altering its logs, status or lock."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'.run.lock').exists():
        raise RuntimeError('OUTPUT_ALREADY_LOCKED')
    if any(OUT.iterdir()):
        raise RuntimeError('OUTPUT_DIRECTORY_NOT_EMPTY')
    # Exclusive creation also closes the race between two empty-directory checks.
    with (OUT/'.run.lock').open('x', encoding='utf-8') as lock:
        lock.write(stamp())
    if any(path.name != '.run.lock' for path in OUT.iterdir()):
        raise RuntimeError('OUTPUT_DIRECTORY_CHANGED_DURING_LOCK')


def tree_rss(child):
    try:
        process = psutil.Process(child.pid)
        processes = [process]+process.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def stop_child(child):
    """Stop only this guard's child/session, then reap the direct child."""
    if child is None:
        return
    if os.name != 'nt':
        # Popen creates a new session, so this cannot signal the parent notebook.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            process = psutil.Process(child.pid)
            descendants = process.children(recursive=True)
            processes = [process]+descendants
        except psutil.NoSuchProcess:
            processes = []
        for process in processes:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=10)
        if alive:
            raise RuntimeError('CHILD_TREE_DID_NOT_EXIT')
    child.wait(timeout=20)


def validate_outputs():
    """Validate the stable artifact set and bytes produced by this claimed run."""
    if (OUT/'failure.json').exists():
        raise ValueError('Runner left failure.json')
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    if not isinstance(result, dict) or result.get('status') != 'COMPLETE':
        raise ValueError('Runner result is not COMPLETE')
    manifest = json.loads((OUT/'artifacts.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), list) or not manifest['files']:
        raise ValueError('Invalid artifact manifest structure')
    root = OUT.resolve()
    listed = set()
    for entry in manifest['files']:
        if not isinstance(entry, dict):
            raise ValueError('Artifact entry must be an object')
        name = entry.get('path')
        if not isinstance(name, str) or not name or '\\' in name:
            raise ValueError('Artifact path must be a relative POSIX path')
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name or ':' in name:
            raise ValueError('Unsafe artifact path: '+name)
        if relative.name in DYNAMIC_NAMES or relative.name.endswith('.tmp') or name in listed:
            raise ValueError('Duplicate or dynamic artifact: '+name)
        path = (OUT/name).resolve(strict=True)
        path.relative_to(root)
        if not path.is_file():
            raise ValueError('Artifact is not a file: '+name)
        size, digest = entry.get('bytes'), entry.get('sha256')
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('Invalid artifact size: '+name)
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid artifact digest: '+name)
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Artifact bytes changed: '+name)
        listed.add(name)
    actual = {
        path.relative_to(OUT).as_posix() for path in OUT.rglob('*')
        if path.is_file() and path.name not in DYNAMIC_NAMES and not path.name.endswith('.tmp')
    }
    if 'result.json' not in listed or listed != actual:
        raise ValueError('Artifact manifest does not cover the stable output files')


def main():
    claim_output()
    started = time.monotonic()
    peak = 0
    child = None
    cause = None
    detail = None
    budget = None
    rss_limit = None
    validated = False
    child_stopped = False
    try:
        protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
        budget = float(protocol['work_budget_seconds'])
        rss_limit = int(protocol['rss_limit_bytes'])
        if not math.isfinite(budget) or budget <= 0 or rss_limit <= 0:
            raise ValueError('Work and RSS budgets must be positive and finite')
        env = os.environ.copy()
        env['NTL_OUTPUT_DIR'] = str(OUT)
        env['PYTHONUTF8'] = '1'
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '2'
        log = OUT/'training.log'
        with log.open('w', encoding='utf-8') as stream, log.open('r', encoding='utf-8', errors='replace') as mirror:
            child = subprocess.Popen(
                [sys.executable, '-u', str(HERE/'runner.py')],
                stdout=stream, stderr=subprocess.STDOUT, env=env,
                start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            while True:
                elapsed = time.monotonic()-started
                if elapsed >= budget:
                    cause = 'WALL_BUDGET_EXCEEDED'
                    break
                if child.poll() is not None:
                    break
                rss = tree_rss(child)
                peak = max(peak, rss)
                save('guard.json', {'status':'RUNNING', 'observed_at':stamp(), 'elapsed_seconds':elapsed,
                     'rss_bytes':rss, 'peak_rss_bytes':peak, 'wall_seconds':budget, 'rss_limit_bytes':rss_limit})
                text = mirror.read()
                if text:
                    print(text, end='', flush=True)
                if rss > rss_limit:
                    cause = 'RSS_BUDGET_EXCEEDED'
                    break
                # Wake promptly when the child exits; a polling sleep would add
                # up to two seconds and could misclassify an on-time completion.
                try:
                    child.wait(timeout=min(2.0, max(0.001, budget-(time.monotonic()-started))))
                except subprocess.TimeoutExpired:
                    pass
            if cause is not None:
                stop_child(child)
                child_stopped = True
            print(mirror.read(), end='', flush=True)
        if cause is None and child.returncode != 0:
            cause = 'CHILD_FAILED'
        if cause is None:
            try:
                validate_outputs()
                validated = True
            except Exception:
                cause = 'OUTPUT_VALIDATION_FAILED'
                detail = traceback.format_exc()
    except BaseException:
        cause = cause or 'GUARD_FAILED'
        detail = traceback.format_exc()
    finally:
        if child is not None and not child_stopped and (cause is not None or child.poll() is None):
            try:
                stop_child(child)
            except BaseException:
                cause = cause or 'CHILD_CLEANUP_FAILED'
                detail = (detail or '')+'\nCleanup error:\n'+traceback.format_exc()
    success = cause is None and validated and child is not None and child.returncode == 0
    exit_code = child.returncode if child is not None else None
    terminal = {'status':'COMPLETE' if success else 'ERROR', 'observed_at':stamp(),
                'child_exit':exit_code, 'termination_reason':cause,
                'elapsed_seconds':time.monotonic()-started, 'peak_rss_bytes':peak,
                'wall_seconds':budget, 'rss_limit_bytes':rss_limit, 'artifacts_verified':validated}
    if detail:
        terminal['traceback'] = detail
    save('guard.json', terminal)
    if not success:
        if not (OUT/'failure.json').exists():
            save('failure.json', {'observed_at':stamp(), 'cause':cause or 'CHILD_FAILED',
                                 'exit_code':exit_code, 'traceback':detail})
        save('state.json', {'status':'ERROR', 'observed_at':stamp(), 'cause':cause or 'CHILD_FAILED'})
        raise RuntimeError(cause or f'CPU experiment failed ({exit_code})')


if __name__ == '__main__':
    main()
