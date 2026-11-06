"""Opt-in read-only numerical snapshots for the fixed ATLAS20 source.

Capture commits establish byte integrity only, never clean OOF or model coverage.
No model is loaded here; the original pipeline owns all computation and clocks.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from full_stack_cache import CacheWriter, LABELS


class CaptureFailure(SystemExit):
    """Do not let the pipeline's optional-model except Exception hide IO failure."""


class NullObserver:
    enabled = False

    def capture(self, event, **payload):
        return None

    def return_array(self, event, value):
        return value


class FullStackObserver:
    enabled = True

    @classmethod
    def from_env(cls, component, ids, texts, labels, metadata=None):
        destination = os.environ.get('NTL_CACHE_EXPORT')
        if not destination:
            return NullObserver()
        context_path = os.environ.get('NTL_CACHE_CONTEXT')
        if not context_path:
            raise ValueError('NTL_CACHE_CONTEXT is required when capture is enabled')
        path = Path(context_path)
        context = json.loads(path.read_text(encoding='utf-8'))
        return cls(destination, component, ids, texts, labels, context, metadata,
                   context_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def __init__(self, root, component, ids, texts, labels, context, metadata=None, context_sha256=None):
        if component not in ('main', 'comp') or tuple(labels) != LABELS:
            raise ValueError('Unsupported component or label order')
        if context.get('schema_version') != 1 or context.get('scope') not in (
                'synthetic_math', 'public_sample_engineering', 'exploratory_external'):
            raise ValueError('Explicit supported data scope required; clean OOF is not certified')
        if context['scope'] != 'synthetic_math' and context.get('launch_ready') is not True:
            raise ValueError('Real capture context is not prepared; bind inputs and budget first')
        if not isinstance(context.get('run_id'), str) or not context['run_id']:
            raise ValueError('Explicit run ID required')
        if not isinstance(context.get('required_events'), dict) or not context['required_events'].get(component):
            raise ValueError('Declare required capture events before executing models')
        required = context['required_events'][component]
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required) or 'component_end' not in required:
            raise ValueError('Required events must explicitly include component_end')
        if (metadata or {}).get('smoke') and context['scope'] != 'synthetic_math':
            raise ValueError('Pseudo-feature SMOKE cannot be labeled real model inference')
        self.component = component
        self.context = context
        self.events = []
        self.array_names = []
        self.note_names = []
        self.sequence = 0
        self.finished = False
        self.writer = CacheWriter(root, component, list(ids), list(texts), labels, metadata={
            'run_id': context['run_id'], 'scope': context['scope'],
            'context_sha256': context_sha256, 'context': context,
            'runtime_metadata': metadata or {},
            'scientific_validation_certified': False,
            'clock_policy': 'Original T0 and ATLAS budget unchanged; capture IO can affect elapsed coverage',
        })
        self.capture('data', ids=list(ids), labels=list(labels),
                     text_sha256=[hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts],
                     scope=context['scope'], component=component)

    def _encode(self, value, prefix, counter):
        if isinstance(value, np.ndarray):
            counter[0] += 1
            key = f'{prefix}-array-{counter[0]:04d}'
            self.writer.array(key, value)
            self.array_names.append(key)
            return {'kind': 'array', 'name': key}
        if isinstance(value, np.generic):
            return self._encode(value.item(), prefix, counter)
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError('Capture metadata mapping keys must be strings')
            # A list preserves reduction/member order even if a JSON writer sorts keys.
            return {'kind': 'mapping', 'items': [[k, self._encode(v, prefix, counter)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {'kind': 'sequence', 'items': [self._encode(v, prefix, counter) for v in value]}
        if value is None or isinstance(value, (str, bool, int, float)):
            json.dumps(value, allow_nan=False)
            return {'kind': 'scalar', 'value': value}
        raise TypeError('Unsupported capture value: ' + type(value).__name__)

    def capture(self, event, **payload):
        try:
            return self._capture(event, **payload)
        except Exception as error:
            raise CaptureFailure('Cache capture failed: ' + str(error)) from error

    def _capture(self, event, **payload):
        if self.finished:
            raise ValueError('Capture after component end')
        if not isinstance(event, str) or not event or not event.replace('_', '').isalnum():
            raise ValueError('Invalid event name')
        self.sequence += 1
        key = f'event-{self.sequence:04d}'
        encoded = self._encode(payload, key, [0])
        self.writer.note(key, {'event': event, 'sequence': self.sequence, 'payload': encoded})
        self.note_names.append(key)
        self.events.append(event)
        if event == 'component_end':
            missing = sorted(set(self.context['required_events'][self.component]) - set(self.events))
            if missing:
                raise ValueError('Missing required capture events: ' + ','.join(missing))
            self.writer.note('event-index', {'events': self.note_names, 'names': self.events,
                'scope': self.context['scope'], 'full_model_coverage_verified': False,
                'clean_validation_verified': False,
                'note': 'Validate expected member inventory, ATLAS masks and replay separately.'})
            self.note_names.append('event-index')
            self.commit_receipt = self.writer.finish(self.array_names, self.note_names, metadata={
                'capture_events_complete': True,
                'full_model_coverage_verified': False,
                'clean_validation_verified': False,
            })
            self.finished = True
            print('CACHE_COMPONENT_COMMITTED ' + json.dumps({
                'component': self.component, 'run_id': self.context['run_id'],
                'manifest_sha256': self.commit_receipt['manifest_sha256'],
                'scope': self.context['scope'], 'full_model_coverage_verified': False,
            }, sort_keys=True), flush=True)

    def return_array(self, event, value):
        self.capture(event, value=value)
        return value  # Deliberately the original object, not the saved copy.


def decode_events(snapshot):
    """Recover the exact ordered event inputs from an already verified snapshot."""
    def decode(node):
        kind = node['kind']
        if kind == 'array':
            return snapshot.arrays[node['name']].copy()
        if kind == 'mapping':
            keys = [x[0] for x in node['items']]
            if len(set(keys)) != len(keys):
                raise ValueError('Duplicate event mapping keys')
            return {k: decode(v) for k, v in node['items']}
        if kind == 'sequence':
            return [decode(v) for v in node['items']]
        if kind == 'scalar':
            return node['value']
        raise ValueError('Unknown event encoding')

    index = snapshot.notes['event-index']
    events = []
    for i, name in enumerate(index['events'], 1):
        event = snapshot.notes[name]
        if event['sequence'] != i or event['event'] != index['names'][i-1]:
            raise ValueError('Event order differs')
        events.append((event['event'], decode(event['payload'])))
    return events
