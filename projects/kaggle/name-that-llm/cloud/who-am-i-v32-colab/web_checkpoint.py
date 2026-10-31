"""Validated, atomic stage checkpoints. Never deserialize executable pickle data."""
import hashlib
import json
import os
import random
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def signature(value):
    def normalized(item):
        if isinstance(item, np.ndarray):
            data = item.tolist() if item.dtype.hasobject else hashlib.sha256(item.tobytes(order='C')).hexdigest()
            return {'array': str(item.dtype), 'shape': item.shape, 'data': data}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(k): normalized(v) for k, v in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [normalized(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def capture_rng():
    state = {'python': random.getstate(), 'numpy': np.random.get_state()}
    torch = sys.modules.get('torch')
    if torch is not None:
        state['torch_cpu'] = torch.random.get_rng_state().cpu().numpy()
        if torch.cuda.is_initialized():
            state['torch_cuda'] = [v.cpu().numpy() for v in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch = sys.modules.get('torch')
    if torch is not None and 'torch_cpu' in state:
        torch.random.set_rng_state(torch.from_numpy(state['torch_cpu'].copy()))
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state_all([torch.from_numpy(v.copy()) for v in state['torch_cuda']])


class Checkpoints:
    def __init__(self, root, context, on_replay=None):
        self.root = Path(root) / context
        self.root.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.on_replay = on_replay
        self.replayed_seconds = 0.0

    def clock(self):
        return time.monotonic() + self.replayed_seconds

    def event(self, stage, status, **extra):
        value = {'stage': stage, 'status': status, 'updated_at': time.time(), **extra}
        atomic_json(self.root / 'progress.json', value)
        print('CHECKPOINT ' + json.dumps(value, ensure_ascii=False), flush=True)

    def marker(self, key):
        return self.root / (hashlib.sha256(key.encode()).hexdigest() + '.json')

    def load(self, key):
        marker = self.marker(key)
        if not marker.is_file():
            return None
        try:
            meta = json.loads(marker.read_text(encoding='utf-8'))
            assert meta['schema'] == SCHEMA and meta['context'] == self.context and meta['key'] == key
            assert Path(meta['payload']).name == meta['payload']
            payload = self.root / meta['payload']
            assert file_hash(payload) == meta['sha256']
            with np.load(payload, allow_pickle=False) as archive:
                def decode(node):
                    kind = node['type']
                    if kind == 'array':
                        return archive[node['name']].copy()
                    if kind in ('tuple', 'list'):
                        values = [decode(v) for v in node['items']]
                        return tuple(values) if kind == 'tuple' else values
                    if kind == 'dict':
                        return {k: decode(v) for k, v in node['items'].items()}
                    if kind == 'scalar':
                        return node['value']
                    raise ValueError('Unknown checkpoint value type')
                result = decode(meta['tree'])
            duration = float(meta['seconds'])
            assert np.isfinite(duration) and duration >= 0
        except (OSError, ValueError, KeyError, TypeError, AssertionError, EOFError, zipfile.BadZipFile):
            self.event(key, 'INVALID_RECOMPUTE')
            return None
        previous_rng = capture_rng()
        try:
            restore_rng(result['rng'])
        except Exception:
            restore_rng(previous_rng)
            self.event(key, 'INVALID_RECOMPUTE')
            return None
        self.replayed_seconds += duration
        if self.on_replay:
            self.on_replay(duration)
        self.event(key, 'RESTORED', original_seconds=round(duration, 3))
        return result['value']

    def save(self, key, value, seconds):
        arrays = {}
        def encode(item):
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject:
                    raise TypeError('Object arrays cannot be checkpointed')
                name = 'array_' + str(len(arrays))
                arrays[name] = item
                return {'type': 'array', 'name': name}
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, (tuple, list)):
                return {'type': type(item).__name__, 'items': [encode(v) for v in item]}
            if isinstance(item, dict):
                if not all(isinstance(k, str) for k in item):
                    raise TypeError('Checkpoint dictionary keys must be strings')
                return {'type': 'dict', 'items': {k: encode(v) for k, v in item.items()}}
            if item is None or isinstance(item, (str, int, float, bool)):
                return {'type': 'scalar', 'value': item}
            raise TypeError(f'Unsupported checkpoint value: {type(item).__name__}')
        tree = encode({'value': value, 'rng': capture_rng()})
        temporary = self.root / (uuid.uuid4().hex + '.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_hash(temporary)
        payload = self.root / (digest + '.npz')
        os.replace(temporary, payload)
        atomic_json(self.marker(key), {'schema': SCHEMA, 'context': self.context, 'key': key,
                    'seconds': max(0.0, float(seconds)), 'payload': payload.name,
                    'sha256': digest, 'tree': tree, 'completed_at': time.time()})
        self.event(key, 'SAVED', seconds=round(float(seconds), 3))

    def call(self, key, function):
        cached = self.load(key)
        if cached is not None:
            return cached
        self.event(key, 'RUNNING')
        started = self.clock()
        value = function()
        self.save(key, value, self.clock() - started)
        return value

    def memoized(self, name):
        def decorate(function):
            def wrapped(*args, **kwargs):
                key = name + '/' + signature({'args': args, 'kwargs': kwargs})
                return self.call(key, lambda: function(*args, **kwargs))
            return wrapped
        return decorate
