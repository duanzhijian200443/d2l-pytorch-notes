import json
from pathlib import Path
import tempfile
import unittest

from drive_inputs import plan_inputs, create_aliases, validate_inputs, _relative


def write(root, relative, content='x'):
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


class DriveInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.drive = self.root / 'drive'
        self.drive.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def item(self, relocations):
        return {'ref':'fixture/model','version':1,'target':'models/fixture',
                'legacy':'old/model','relocations':relocations}

    def test_renamed_root_is_used_instead_of_stale_cache(self):
        write(self.drive,'old/model/config.json','old')
        expected = write(self.drive,'new/model/config.json','new')
        manifest = {'inputs':[self.item([{'source':'new/model','destination':'','replace_root':True}])]}
        plan,_ = plan_inputs(manifest,self.drive)
        self.assertEqual(plan[Path('models/fixture/config.json')],expected)

    def test_empty_renamed_model_is_not_silently_filled_from_old_version(self):
        write(self.drive,'old/model/config.json','old')
        (self.drive/'new/model').mkdir(parents=True)
        manifest = {'inputs':[self.item([{'source':'new/model','destination':'','replace_root':True}])]}
        with self.assertRaises(FileNotFoundError):
            plan_inputs(manifest,self.drive)

    def test_split_checkpoint_keeps_required_alias_and_unmoved_files(self):
        sidecar = write(self.drive,'old/model/sidecar.json')
        weights = write(self.drive,'library/renamed-encoder/model.safetensors')
        manifest = {'inputs':[self.item([{'source':'library/renamed-encoder','destination':'deberta-v3-large'}])]}
        plan,_ = plan_inputs(manifest,self.drive)
        aliases = self.root/'aliases'
        create_aliases(plan,aliases)
        self.assertEqual((aliases/'models/fixture/deberta-v3-large/model.safetensors').resolve(),weights)
        self.assertEqual((aliases/'models/fixture/sidecar.json').resolve(),sidecar)
        create_aliases(plan,aliases)  # repeat is idempotent
        self.assertTrue(weights.is_file())

    def test_alias_conflict_does_not_overwrite_or_modify_source(self):
        first=write(self.drive,'one/file'); second=write(self.drive,'two/file','second')
        aliases=self.root/'aliases'
        create_aliases({Path('item/file'):first},aliases)
        with self.assertRaises(RuntimeError):
            create_aliases({Path('item/file'):second},aliases)
        self.assertEqual((aliases/'item/file').read_text(),'x')
        self.assertEqual(second.read_text(),'second')

    def test_directory_link_cannot_redirect_writes_into_drive(self):
        source=write(self.drive,'source.bin')
        aliases=self.root/'aliases'; aliases.mkdir()
        write(aliases,'.atlas20-drive-aliases.json',json.dumps({'owner':'atlas20-drive-v1'}))
        (aliases/'models').symlink_to(self.drive,target_is_directory=True)
        with self.assertRaises(RuntimeError):
            create_aliases({Path('models/accidental.bin'):source},aliases)
        self.assertFalse((self.drive/'accidental.bin').exists())

    def test_missing_indexed_shard_is_rejected(self):
        aliases=self.root/'aliases'
        write(aliases,'models/q/config.json',json.dumps({'model_type':'qwen3'}))
        write(aliases,'models/q/model.safetensors.index.json',json.dumps({'weight_map':{'a':'shard-1.safetensors','b':'shard-2.safetensors'}}))
        write(aliases,'models/q/shard-1.safetensors')
        write(aliases,'competitions/name-that-llm/test.csv','id,text\n1,example\n')
        manifest={'required':[],'model_config_patterns':['models/**/config.json']}
        with self.assertRaisesRegex(FileNotFoundError,'shard-2'):
            validate_inputs(aliases,manifest)
        write(aliases,'models/q/shard-2.safetensors')
        result=validate_inputs(aliases,manifest)
        self.assertEqual(result['indexed_shards'],2)
        self.assertEqual(result['test_rows'],1)
        self.assertFalse(result['all_model_content_hashes_verified'])

    def test_path_escape_and_unowned_directory_rejected(self):
        for bad in ('../outside','/outside'):
            with self.assertRaises(ValueError): _relative(bad)
        aliases=self.root/'aliases'; write(aliases,'user-file.txt','keep')
        with self.assertRaises(RuntimeError): create_aliases({},aliases)
        self.assertEqual((aliases/'user-file.txt').read_text(),'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
