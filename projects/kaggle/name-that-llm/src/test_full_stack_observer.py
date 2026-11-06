"""Pure synthetic tests of non-mutating, ordered observation and failure closure."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from full_stack_cache import LABELS, read_cache
from full_stack_observer import FullStackObserver, NullObserver, CaptureFailure, decode_events


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ntl-observer-test-')
        self.root=Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent,Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('ntl-observer-test-'))
        self.addCleanup(self.temp.cleanup)
        self.ids=['row-a','row-b'];self.texts=['synthetic a','synthetic b']
        self.context={'schema_version':1,'scope':'synthetic_math','run_id':'synthetic',
            'required_events':{'main':['main_members','component_end'],'comp':['member','component_end']}}

    def observer(self):
        return FullStackObserver(self.root,'main',self.ids,self.texts,list(LABELS),self.context)

    def test_disabled_capture_is_identity_and_does_not_create_files(self):
        with patch.dict(os.environ,{},clear=True):
            observer=FullStackObserver.from_env('main',self.ids,self.texts,LABELS)
        self.assertIsInstance(observer,NullObserver)
        matrix=np.eye(18)
        self.assertIs(observer.return_array('style_margin',matrix),matrix)
        observer.capture('anything',array=matrix)
        self.assertFalse(list(self.root.iterdir()))

    def test_enabled_requires_explicit_supported_scope(self):
        with patch.dict(os.environ,{'NTL_CACHE_EXPORT':str(self.root)},clear=True):
            with self.assertRaisesRegex(ValueError,'CONTEXT'):
                FullStackObserver.from_env('main',self.ids,self.texts,LABELS)
        with self.assertRaises(ValueError):
            FullStackObserver(self.root,'main',self.ids,self.texts,LABELS,{**self.context,'scope':'clean_oof'})
        with self.assertRaisesRegex(ValueError,'not prepared'):
            FullStackObserver(self.root,'main',self.ids,self.texts,LABELS,{**self.context,'scope':'public_sample_engineering','launch_ready':False})

    def test_snapshot_preserves_dtype_order_values_and_return_identity(self):
        observer=self.observer()
        matrix=np.arange(36,dtype=np.float32).reshape(2,18)
        original=matrix.copy()
        self.assertIs(observer.return_array('style_margin',matrix),matrix)
        observer.capture('main_members',members={'z_first':matrix,'a_second':matrix.astype(np.float64)},
                         options={'tuple':(np.int64(3),True,None)})
        matrix[:]=-999
        observer.capture('component_end',output_path='synthetic.parquet')
        snapshot=read_cache(self.root,'main',self.ids,self.texts,LABELS)
        events=decode_events(snapshot)
        self.assertEqual([x[0] for x in events],['data','style_margin','main_members','component_end'])
        self.assertEqual(events[0][1]['ids'],self.ids)
        np.testing.assert_array_equal(events[1][1]['value'],original)
        self.assertEqual(list(events[2][1]['members']),['z_first','a_second'])
        self.assertEqual(events[2][1]['members']['z_first'].dtype,np.dtype('float32'))
        self.assertEqual(events[2][1]['members']['a_second'].dtype,np.dtype('float64'))
        self.assertFalse(snapshot.finish_metadata['clean_validation_verified'])
        self.assertFalse(snapshot.finish_metadata['full_model_coverage_verified'])
        self.assertNotIn('synthetic a',(self.root/'main'/'manifest.json').read_text())

    def test_missing_stage_cannot_commit(self):
        observer=self.observer()
        with self.assertRaisesRegex(CaptureFailure,'Missing required'):
            observer.capture('component_end')
        self.assertFalse((self.root/'main'/'manifest.json').exists())
        with self.assertRaises(Exception):
            read_cache(self.root,'main',self.ids,self.texts,LABELS)

    def test_storage_failure_is_not_swallowed_by_optional_model_handler(self):
        observer=self.observer()
        caught_by_optional=False
        with self.assertRaises(CaptureFailure):
            try:
                observer.capture('main_members',array=np.array([np.nan]))
            except Exception:
                caught_by_optional=True
        self.assertFalse(caught_by_optional)
        self.assertFalse((self.root/'main'/'manifest.json').exists())


if __name__=='__main__':unittest.main()
