"""EXP044 CPU supervisor: cumulative wall budget, process-tree RSS, explicit resume."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import psutil

HERE = Path(__file__).resolve().parent
EXCLUDED = {'artifacts.json', '.guard.lock', '.claim.lock', 'guard.json', 'state.json'}
spec = importlib.util.spec_from_file_location('exp044_runner_guard_helpers', HERE / 'runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def tree_rss(pid):
    try:
        process = psutil.Process(pid)
        processes = [process] + process.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def charge_previous(ledger, now):
    """A lost guard is charged through now, including downtime, capped at budget.

    This conservative reservation cannot buy an extra budget after a hard kill.
    A normally terminated attempt is charged its recorded elapsed runtime only.
    """
    used = 0.
    for attempt in ledger['attempts']:
        charge = attempt.get('charged_seconds')
        if charge is None:
            remaining = max(0., ledger['hard_seconds'] - used)
            charge = min(now - attempt['started_wall'], remaining) if now >= attempt['started_wall'] else remaining
            attempt['charged_seconds'] = charge
            attempt['status'] = 'LOST_GUARD_CHARGED_CONSERVATIVELY'
            attempt['charge_rule'] = 'wall through explicit resume, including downtime, capped at remaining budget'
        r.need(isinstance(charge, (int, float)) and not isinstance(charge, bool) and math.isfinite(charge) and charge >= 0, 'Invalid prior budget charge')
        used += charge
    return used


def confirm_inactive(output, old_ticket):
    terminal_path = output / 'attempts' / old_ticket['attempt'] / 'guard.json'
    terminal = r.read(terminal_path) if terminal_path.exists() else {}
    terminal_proof = (terminal.get('nonce') == old_ticket['nonce'] and terminal.get('tree_stopped') is True
                      and terminal.get('status') in ('ERROR', 'COMPLETE'))
    try:
        alive = r.process_live(old_ticket['process'])
    except ValueError:
        r.need(terminal_proof, 'Unknown prior host/boot; terminal tree-stop proof required')
        alive = False
    r.need(not alive, 'Guard process is still active; resume rejected')
    # Terminal guard cleanup covers descendants across hosts. On the same live
    # host, also inspect every recorded coordinator/worker before permitting reuse.
    for path in list(output.glob('attempts/*/process.json')) + list(output.glob('jobs/*/attempt-*/process.json')) + list(output.glob('selftests/*/jobs/*/attempt-*/process.json')):
        try:
            r.need(not r.process_live(r.read(path)), 'Prior coordinator/worker is still active')
        except ValueError as error:
            if 'across host/boot' not in str(error) or not terminal_proof:
                raise


@contextmanager
def claim_mutex(output):
    """Kernel-held nonblocking lock releases on process death; never stale PID trust."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.claim.lock').open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def claim(output, payload, resume):
    with claim_mutex(output):
        return claim_locked(output, payload, resume)


def claim_locked(output, payload, resume):
    config = r.read(payload / 'config.json')
    protocol = r.read(payload / 'protocol.json')
    r.need(protocol['work_budget_seconds'] == r.HARD_SECONDS and protocol['rss_limit_bytes'] == r.RSS_LIMIT, 'Guard protocol budget changed')
    r.need(config['budget']['hard_seconds'] == r.HARD_SECONDS and config['budget']['rss_limit_bytes'] == r.RSS_LIMIT, 'Config guard budget mismatch')
    r.need(config['input_hashes']['guard.py'] == r.sha(__file__), 'Executing guard hash mismatch')
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.guard.lock'
    if resume:
        r.need(lock.exists() and (output / 'budget.json').exists() and (output / 'run.json').exists(), 'Resume has no original lock/budget/run identity')
        old_ticket = r.read(lock)
        confirm_inactive(output, old_ticket)
        r.need(not ((output / 'result.json').exists() and r.read(output / 'result.json').get('status') == 'COMPLETE'), 'Completed run cannot be resumed')
        # Validate the complete saved stable tree before any attempt can mutate it.
        if (output / 'artifacts.json').exists():
            previous = r.read(output / 'artifacts.json')
            r.validate_inventory(output, previous['files'], EXCLUDED)
        ledger = r.read(output / 'budget.json')
        r.need(ledger['run_id'] == config['run_id'] and ledger['config_sha256'] == r.sha(payload / 'config.json')
               and ledger['hard_seconds'] == r.HARD_SECONDS and ledger['protocol_sha256'] == r.sha(payload / 'protocol.json'), 'Resume budget/config/protocol identity mismatch')
        used = charge_previous(ledger, time.time())
        r.need(used < r.HARD_SECONDS, 'Cumulative wall budget exhausted')
        retired = output / 'attempts' / old_ticket['attempt'] / 'retired-lock.json'
        r.need(not retired.exists(), 'Prior lock already retired; ambiguous resume')
        lock.replace(retired)
    else:
        r.need(not any(p.name != '.claim.lock' for p in output.iterdir()), 'Existing output requires explicit --resume')
        ledger = {'run_id': config['run_id'], 'config_sha256': r.sha(payload / 'config.json'),
                  'protocol_sha256': r.sha(payload / 'protocol.json'), 'hard_seconds': r.HARD_SECONDS, 'attempts': []}
        used = 0.
    attempt_id = f'attempt-{len(ledger["attempts"])+1:04d}'
    ticket = {'attempt': attempt_id, 'nonce': uuid.uuid4().hex, 'process': r.process_record()}
    # Exclusive creation prevents two resumers from claiming one checkpoint.
    with lock.open('x', encoding='utf-8') as stream:
        import json
        json.dump(ticket, stream)
    attempt_dir = output / 'attempts' / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    entry = {'attempt': attempt_id, 'nonce': ticket['nonce'], 'status': 'RUNNING',
             'started_wall': time.time(), 'charged_seconds': None, 'used_before_seconds': used}
    ledger['attempts'].append(entry)
    r.save(output / 'budget.json', ledger)
    return ledger, ticket, used


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results')))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    payload, output = args.payload.resolve(), args.output.resolve()
    ledger, ticket, used = claim(output, payload, args.resume)
    started = time.monotonic()
    remaining = r.HARD_SECONDS - used
    attempt_dir = output / 'attempts' / ticket['attempt']
    child = None
    reason, detail, peak, tree_stopped = None, None, 0, False
    try:
        command = [sys.executable, '-u', str(payload / 'runner.py'), '--payload', str(payload), '--output', str(output), '--_ticket', ticket['nonce']]
        if args.resume:
            command.append('--resume')
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '2'
        with (attempt_dir / 'training.log').open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            r.save(attempt_dir / 'process.json', r.process_record(child.pid))
            last_save, last_print = 0., -30.
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= remaining:
                    reason = 'CUMULATIVE_WALL_BUDGET_EXCEEDED'
                    break
                if child.poll() is not None:
                    break
                rss = tree_rss(os.getpid())
                peak = max(peak, rss)
                if rss > r.RSS_LIMIT:
                    reason = 'RSS_BUDGET_EXCEEDED'
                    break
                if elapsed - last_save >= 2:
                    r.save(output / 'guard.json', {'status': 'RUNNING', 'nonce': ticket['nonce'], 'attempt': ticket['attempt'],
                            'attempt_elapsed_seconds': elapsed, 'cumulative_elapsed_seconds': used+elapsed,
                            'rss_bytes': rss, 'peak_rss_bytes': peak, 'remaining_at_start_seconds': remaining})
                    last_save = elapsed
                if elapsed - last_print >= 30:
                    state_path = output / 'state.json'
                    state = r.read(state_path) if state_path.exists() else {}
                    print(json.dumps({'status': 'RUNNING', 'attempt': ticket['attempt'],
                        'cumulative_seconds': round(used+elapsed, 2), 'rss_bytes': rss, 'peak_rss_bytes': peak,
                        'completed': state.get('completed', []), 'active': state.get('active', []),
                        'phase': state.get('status', 'PREFLIGHT_OR_SELFTEST')}), flush=True)
                    last_print = elapsed
                try:
                    child.wait(timeout=min(.25, max(.01, remaining-elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            if reason is None and child.returncode != 0:
                reason = 'COORDINATOR_FAILED'
            if reason is None:
                result = r.read(output / 'result.json')
                r.need(result['status'] == 'COMPLETE' and result['trained_svms'] == 20, 'Coordinator result incomplete')
                identity = r.read(output / 'run.json')['identity']
                r.need(result['identity'] == identity, 'Result identity mismatch')
                completed = r.find_completed(output, identity, [(f, a) for f in range(5) for a in r.ARMS])
                r.need(len(completed) == 10, 'Not all fold jobs sealed')
    except BaseException:
        reason = reason or 'GUARD_FAILED'
        detail = traceback.format_exc()
    finally:
        try:
            if child is not None:
                if child.poll() is None:
                    r.stop_tree(child.pid)
                child.wait(timeout=20)
            # Orphans from an unexpected coordinator death are found from recorded
            # PIDs, never from name matching or another run's process list.
            for path in list(output.glob('jobs/*/attempt-*/process.json')) + list(output.glob('selftests/*/jobs/*/attempt-*/process.json')):
                record = r.read(path)
                try:
                    if r.process_live(record):
                        r.stop_tree(record['pid'])
                except ValueError:
                    # Previous host/boot identities were admitted only by claim's
                    # terminal tree-stop proof; never target their recycled PIDs.
                    if (record['host'] == r.platform.node()
                            and abs(record['boot_time'] - psutil.boot_time()) < 2):
                        raise
            tree_stopped = True
        except BaseException:
            reason = reason or 'TREE_CLEANUP_FAILED'
            detail = (detail or '') + '\n' + traceback.format_exc()
    elapsed = time.monotonic() - started
    if elapsed + used >= r.HARD_SECONDS:
        reason = reason or 'CUMULATIVE_WALL_BUDGET_EXCEEDED'
    terminal = {'status': 'COMPLETE' if reason is None else 'ERROR', 'attempt': ticket['attempt'], 'nonce': ticket['nonce'],
                'child_exit': child.returncode if child else None, 'termination_reason': reason, 'tree_stopped': tree_stopped,
                'attempt_elapsed_seconds': elapsed, 'cumulative_elapsed_seconds': used + elapsed,
                'hard_seconds': r.HARD_SECONDS, 'rss_limit_bytes': r.RSS_LIMIT, 'peak_rss_bytes': peak}
    if detail:
        terminal['traceback'] = detail
    ledger['attempts'][-1].update(status=terminal['status'], charged_seconds=elapsed, finished_wall=time.time())
    r.save(output / 'budget.json', ledger)
    r.save(attempt_dir / 'guard.json', terminal)
    r.save(output / 'guard.json', terminal)
    if reason:
        r.save(attempt_dir / 'failure.json', terminal)
        r.save(output / 'state.json', {'status': 'ERROR', 'attempt': ticket['attempt'], 'cause': reason})
    # Includes all old attempts and sealed job logs. Only root live-control files
    # and this manifest itself are outside the stable byte inventory.
    r.save(output / 'artifacts.json', {'run_id': ledger['run_id'], 'status': terminal['status'],
            'files': r.inventory(output, EXCLUDED)})
    print(json.dumps({'status': terminal['status'], 'attempt': ticket['attempt'], 'cause': reason,
                      'cumulative_seconds': terminal['cumulative_elapsed_seconds'], 'tree_stopped': tree_stopped}), flush=True)
    if reason:
        raise RuntimeError(reason)


if __name__ == '__main__':
    main()
