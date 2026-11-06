"""Offline packaging checks with a clearly synthetic CPU receipt; no uploads."""
import ast
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_capture as build


class PackagingTest(unittest.TestCase):
    def manifest(self, path, status='PASS'):
        config=json.loads((build.HERE/'preflight_config.json').read_text())
        value={'schema_version':1,'status':status,'synthetic_test_receipt':True,
               'attachment_bindings':config['attachment_bindings'],
               'preflight_sha256':build.sha(build.HERE/'preflight.py'),
               'config_sha256':build.sha(build.HERE/'preflight_config.json'),
               'environment':{'versions':{'numpy':'TEST_ONLY'}}}
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_failed_cpu_receipt_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ntl-029-package-') as name:
            root=Path(name);receipt=root/'receipt.json';self.manifest(receipt,'FAILED')
            with patch.object(sys,'argv',['prepare_capture','--preflight-manifest',str(receipt),'--destination',str(root/'out')]):
                with self.assertRaises(AssertionError):build.main()
            self.assertFalse((root/'out').exists())

    def test_only_blinded_panel_and_pinned_code_packaged(self):
        with tempfile.TemporaryDirectory(prefix='ntl-029-package-') as name:
            root=Path(name);receipt=root/'receipt.json';self.manifest(receipt)
            out=root/'out'
            # This test isolates packaging; fake receipts are rejected by the real validator.
            with patch.object(build,'validate_preflight'), patch.object(sys,'argv',['prepare_capture','--preflight-manifest',str(receipt),'--destination',str(out)]), contextlib.redirect_stdout(io.StringIO()):
                build.main()
            source=ast.parse((out/'kernel/capture.py').read_text())
            assignment=next(node for node in source.body if isinstance(node,ast.Assign)
                            and any(isinstance(t,ast.Name) and t.id=='blob' for t in node.targets))
            raw=base64.b64decode(assignment.value.args[0].value)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                names=archive.namelist()
                self.assertFalse(any('labels-local-only' in n or n.endswith('evaluation.csv') for n in names))
                self.assertEqual(archive.read('panel/test.csv'),(build.EXP/'panel/test.csv').read_bytes())
                self.assertEqual(archive.read('panel/test.csv').splitlines()[0],b'id,text')
                self.assertTrue(json.loads(archive.read('preflight_manifest.json'))['synthetic_test_receipt'])
            metadata=json.loads((out/'kernel/kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private']);self.assertTrue(metadata['enable_gpu'])
            self.assertFalse(metadata['enable_internet'])
            self.assertEqual(sum(len(metadata[k]) for k in ('competition_sources','dataset_sources','kernel_sources','model_sources')),31)
            payload=out/'payload';plan=json.loads((payload/'plan.json').read_text())
            (payload/'labels-local-only.csv').write_text('SHOULD_NOT_UPLOAD',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Undeclared'):
                build.runner.verify_payload(payload,plan)


if __name__=='__main__':unittest.main()
