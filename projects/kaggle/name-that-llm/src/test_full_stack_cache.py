"""Small integrity/rejection tests; no model import, fit, or network operation."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from full_stack_cache import CacheError, CacheWriter, LABELS, NullWriter, read_cache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="full-stack-cache-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = ["id-a", "id-b", "id-c"]
        self.texts = ["PRIVATE-TEXT-A-DO-NOT-PERSIST", "PRIVATE-TEXT-B-DO-NOT-PERSIST", ""]

    def writer(self, component="model", **kwargs):
        return CacheWriter(self.root, component, self.ids, self.texts, **kwargs)

    def read(self, component="model", **kwargs):
        return read_cache(self.root, component, self.ids, self.texts, **kwargs)

    def committed(self):
        writer = self.writer(metadata={"checkpoint": "frozen"})
        writer.array("scores", np.arange(54, dtype=np.float32).reshape(3, 18))
        writer.note("routing", {"groups": [1, 2, 3]})
        return writer.finish(["scores"], ["routing"], metadata={"subset": "small"})

    def manifest(self):
        return json.loads((self.root / "model/manifest.json").read_text(encoding="utf-8"))

    def rewrite_manifest(self, value):
        (self.root / "model/manifest.json").write_text(json.dumps(value), encoding="utf-8")

    def test_roundtrip_preserves_dtype_shape_values_and_snapshot(self):
        metadata = {"checkpoint": ["original"]}
        writer = self.writer(metadata=metadata)
        original = np.arange(24, dtype=">f4").reshape(4, 6)[:, ::2]
        frozen = original.copy()
        writer.array("features", original, metadata={"axis": "arbitrary"})
        original[:] = -999
        metadata["checkpoint"].append("mutated")
        writer.array("scalar", np.array(True))
        writer.array("empty", np.empty((0, 2), dtype=np.int16))
        note = {"threshold": [.3]}
        writer.note("grouping", note)
        note["threshold"][0] = .9
        receipt = writer.finish(["features", "scalar", "empty"], ["grouping"])
        snapshot = self.read(manifest_sha256=receipt["manifest_sha256"])
        actual = snapshot.arrays["features"]
        self.assertEqual(actual.dtype.str, ">f4")
        self.assertEqual(actual.shape, frozen.shape)
        self.assertEqual(actual.tobytes(), frozen.tobytes())
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(snapshot.arrays["scalar"].shape, ())
        self.assertEqual(snapshot.arrays["scalar"].dtype, np.dtype(bool))
        self.assertEqual(snapshot.arrays["empty"].shape, (0, 2))
        self.assertEqual(snapshot.metadata, {"checkpoint": ["original"]})
        self.assertEqual(snapshot.notes["grouping"], {"threshold": [.3]})
        self.assertEqual(receipt["commit_scope"], "component_bytes_only")
        for path in (self.root / "model").iterdir():
            self.assertNotIn(self.texts[0].encode(), path.read_bytes())

    def test_input_order_text_fingerprints_and_fixed_labels_are_bound(self):
        self.committed()
        for ids, texts, labels in ((self.ids[::-1], self.texts[::-1], LABELS),
                                    (self.ids, ["changed", *self.texts[1:]], LABELS),
                                    (self.ids, self.texts, LABELS[::-1])):
            with self.subTest(ids=ids, labels=labels), self.assertRaises(CacheError):
                read_cache(self.root, "model", ids, texts, labels)
        for ids in (["same", "same", "third"], [1, 2, 3], ["", "b", "c"]):
            with self.assertRaises(CacheError):
                CacheWriter(self.root, "invalid", ids, self.texts)
        self.assertFalse((self.root / "invalid").exists())

    def test_existing_component_duplicate_and_cross_kind_names_never_overwrite(self):
        writer = self.writer()
        receipt = writer.array("Scores", np.array([1.], dtype=np.float64))
        original = (writer.directory / receipt["path"]).read_bytes()
        for action in (lambda: writer.array("Scores", np.array([2.])),
                       lambda: writer.array("scores", np.array([2.])),
                       lambda: writer.note("Scores", {}), lambda: self.writer()):
            with self.assertRaises(CacheError):
                action()
        self.assertEqual((writer.directory / receipt["path"]).read_bytes(), original)

    def test_unsafe_names_and_paths_are_rejected(self):
        writer = self.writer()
        for name in ("../outside", "a/b", "a\\b", "C:bad", ".", "..", "", "trail.", "CON", "nul.txt"):
            with self.subTest(name=name):
                with self.assertRaises(CacheError):
                    CacheWriter(self.root, name, self.ids, self.texts)
                with self.assertRaises(CacheError):
                    writer.array(name, np.array([1]))
                with self.assertRaises(CacheError):
                    writer.note(name, {})
        self.assertEqual(list(writer.directory.iterdir()), [])

    def test_nonfinite_object_string_arrays_and_nonjson_notes_are_rejected(self):
        writer = self.writer()
        for array in (np.array([np.nan]), np.array([np.inf]), np.array([complex(1, np.inf)]),
                      np.array([{}], dtype=object), np.array(["text"]), np.array([(1,)], dtype=[("field", "i4")])):
            with self.subTest(dtype=array.dtype), self.assertRaises(CacheError):
                writer.array("bad", array)
        for metadata in ({"bad": float("nan")}, {1: "bad"}, {"bad": object()}):
            with self.assertRaises(CacheError):
                writer.note("bad", metadata)
        self.assertEqual(list(writer.directory.iterdir()), [])

    def test_unfinished_missing_required_and_closed_writers_are_rejected(self):
        writer = self.writer()
        writer.array("scores", np.zeros((3, 18)))
        with self.assertRaises(CacheError):
            self.read()
        with self.assertRaises(CacheError):
            writer.finish(["missing"])
        self.assertFalse((writer.directory / "manifest.json").exists())
        writer.close()
        for action in (lambda: writer.array("later", np.array([1.])), lambda: writer.note("later", {}),
                       lambda: writer.finish(["scores"])):
            with self.assertRaises(CacheError):
                action()

    def test_finish_is_once_and_context_exit_does_not_commit(self):
        with self.writer() as writer:
            writer.array("scores", np.ones((3, 18)))
            writer.finish(["scores"])
        with self.assertRaises(CacheError):
            writer.finish(["scores"])
        with self.writer("unfinished") as pending:
            pending.array("scores", np.ones(1))
        with self.assertRaises(CacheError):
            self.read("unfinished")

    def test_tampering_missing_and_extra_files_are_rejected(self):
        receipt = self.committed()
        path = self.root / "model/array-scores.npy"
        original = path.read_bytes()
        for content in (original + b"extra", original[:-1]):
            path.write_bytes(content)
            with self.assertRaises(CacheError):
                self.read()
        path.write_bytes(original)
        path.unlink()
        with self.assertRaises(CacheError):
            self.read()
        path.write_bytes(original)
        extra = self.root / "model/unlisted.txt"; extra.write_text("extra", encoding="utf-8")
        with self.assertRaises(CacheError):
            self.read()
        extra.unlink()
        self.read(manifest_sha256=receipt["manifest_sha256"])

    def test_finish_checks_existing_bytes_before_committing(self):
        writer = self.writer()
        writer.array("scores", np.arange(3))
        (writer.directory / "array-scores.npy").write_bytes(b"truncated")
        with self.assertRaises(CacheError):
            writer.finish(["scores"])
        self.assertFalse((writer.directory / "manifest.json").exists())

    def test_manifest_schema_component_shape_dtype_and_names_are_checked(self):
        self.committed()
        base = self.manifest()
        mutations = []
        changed = copy.deepcopy(base); changed["state"] = "WRITING"; mutations.append(changed)
        changed = copy.deepcopy(base); changed["component"] = "other"; mutations.append(changed)
        changed = copy.deepcopy(base); changed["extra"] = True; mutations.append(changed)
        changed = copy.deepcopy(base); changed["arrays"][0]["shape"] = [54]; mutations.append(changed)
        changed = copy.deepcopy(base); changed["arrays"][0]["dtype"] = "<f8"; mutations.append(changed)
        changed = copy.deepcopy(base); changed["arrays"][0]["path"] = "../outside.npy"; mutations.append(changed)
        changed = copy.deepcopy(base); changed["arrays"].append(changed["arrays"][0]); mutations.append(changed)
        changed = copy.deepcopy(base); changed["identity"]["ids"].reverse(); mutations.append(changed)
        for changed in mutations:
            self.rewrite_manifest(changed)
            with self.subTest(manifest=changed), self.assertRaises(CacheError):
                self.read()

    def test_external_manifest_anchor_rejects_coordinated_metadata_rewrite(self):
        receipt = self.committed()
        changed = self.manifest(); changed["metadata"] = {"checkpoint": "different"}
        self.rewrite_manifest(changed)
        with self.assertRaisesRegex(CacheError, "external receipt"):
            self.read(manifest_sha256=receipt["manifest_sha256"])

    def test_reader_never_loads_object_pickle_even_with_updated_file_hash(self):
        self.committed()
        buffer = io.BytesIO()
        np.save(buffer, np.array([{"object": True}], dtype=object), allow_pickle=True)
        content = buffer.getvalue()
        (self.root / "model/array-scores.npy").write_bytes(content)
        manifest = self.manifest()
        manifest["arrays"][0].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(CacheError, "without pickle"):
            self.read()

    def test_reader_checks_finite_values_even_after_hash_rewrite(self):
        self.committed()
        buffer = io.BytesIO()
        values = np.zeros((3, 18), dtype=np.float32); values[0, 0] = np.nan
        np.save(buffer, values, allow_pickle=False)
        content = buffer.getvalue()
        (self.root / "model/array-scores.npy").write_bytes(content)
        manifest = self.manifest()
        manifest["arrays"][0].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(CacheError, "nonfinite"):
            self.read()

    def test_required_entries_and_note_bytes_are_checked(self):
        self.committed()
        with self.assertRaises(CacheError):
            self.read(required_arrays=["absent"])
        with self.assertRaises(CacheError):
            self.read(required_notes=["absent"])
        (self.root / "model/note-routing.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(CacheError):
            self.read()

    def test_null_writer_does_not_validate_allocate_or_claim_committed(self):
        target = self.root / "disabled-root"
        with NullWriter(target, "any", self.ids, self.texts) as writer:
            self.assertIsNone(writer.array("anything", object()))
            self.assertIsNone(writer.note("anything", object()))
            self.assertEqual(writer.finish(["absent"])["state"], "DISABLED")
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
