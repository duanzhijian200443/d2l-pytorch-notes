"""Read-only input checks and a tiny two-arm fold regression for EXP052."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import train_volume as volume

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "cloud/format120-044/payload"
BASE = ROOT / "cloud/luna-terra120-042/payload"


def table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class VolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("format_for_test", OLD / "runner.py")
        cls.fmt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fmt)
        cls.legacy, cls.core, _, _ = cls.fmt.modules(OLD)
        cls.gate = volume.read(HERE / "metrics/data-gate.json")

    def test_frozen_four_class_data_and_weight_mass(self):
        frames = [table(BASE / name) for name in ("train.csv", "folds.csv", "extra.csv")]
        frames.extend(table(HERE / "data" / (arm + ".csv")) for arm in volume.ARMS)
        config = {"input_hashes": {name: row["sha256"] for name, row in self.gate["outputs"].items()},
                  "holdout_prompt_group_hashes": volume.read(BASE / "config.json")["holdout_prompt_group_hashes"]}
        volume.validate_volume(frames, self.legacy, self.gate, config)
        train, folds, extra, small, large = frames
        indices = {v: i for i, v in enumerate(self.legacy.LABELS)}
        for fold in range(5):
            train_y = train.model[folds.fold.ne(fold)].map(indices).to_numpy()
            base_y = extra.model.map(indices).to_numpy()
            for added, size in [(small, 30), (large, 60)]:
                new_y = added.model.map(indices).to_numpy()
                original, expanded, audit = self.legacy.fixed_weights(train_y, base_y, new_y)
                self.assertTrue(np.array_equal(original, expanded[:len(original)]))
                expected = .5 * np.bincount(train_y)[new_y] / np.bincount(base_y)[new_y]
                np.testing.assert_array_equal(expanded[len(original):], expected)
                self.assertEqual({self.legacy.LABELS[r["class_index"]]: r["added_rows"] for r in audit if r["added_rows"]},
                                 dict.fromkeys(volume.NEW_LABELS, size))

    def test_original_four_answer_fit_and_routing_on_tiny_fold(self):
        config, old_frames, legacy, core, identity = self.fmt.synthetic_setup(OLD)
        small = pd.DataFrame([
            {"prompt_id": f"s{p}", "model": model,
             "text": f"Small test subject {p}, distinctive phrase {model}, response from voice {i}."}
            for p in range(2) for i, model in enumerate(volume.NEW_LABELS)
        ])
        extra = small.copy()
        extra["prompt_id"] += "-additional"
        extra["text"] += " Additional synthetic training material."
        large = pd.concat([small, extra], ignore_index=True)
        frames = [*old_frames[:3], small, large]
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            volume.save(output / "identity.json", identity)
            for arm, selected in zip(volume.ARMS, [small, large]):
                directory = output / "jobs" / ("f0-" + arm)
                directory.mkdir(parents=True)
                (directory / "training.log").touch()
                prototype = (self.fmt, legacy, core, config, frames, identity)
                args = SimpleNamespace(output=output, payload=OLD, fold=0, arm=arm)
                with patch.object(volume, "preflight", return_value=prototype):
                    volume.worker(args)
                self.fmt.seal_job(directory, identity, 0, arm)
                serial = self.fmt.legacy_reference(config, [*frames[:3], selected, selected], legacy, core, arm)
                for file, reference in serial.items():
                    with np.load(directory / file, allow_pickle=False) as actual:
                        np.testing.assert_allclose(actual["scores"], reference["scores"], rtol=0, atol=1e-12)
                        np.testing.assert_array_equal(actual["centred_mask"], reference["centred_mask"])
                        self.assertEqual(actual["arm"].item(), arm)
                weights = volume.read(directory / "weights.json")["classes"]
                self.assertEqual({r["label"]: r["added_rows"] for r in weights if r["added_rows"]},
                                 dict.fromkeys(volume.NEW_LABELS, len(selected) // 4))

    def test_reject_missing_fourth_response_and_holdout(self):
        frame = table(HERE / "data/four120.csv")
        with self.assertRaisesRegex(ValueError, "count"):
            volume.validate_added(frame.iloc[:-1], 120, self.legacy, set(), [])
        held = [self.legacy.text_hash(self.legacy.normalize(frame.base_prompt.iloc[0]))]
        with self.assertRaisesRegex(ValueError, "Held-out"):
            volume.validate_added(frame, 120, self.legacy, set(), held)

    def test_limits(self):
        volume.enforce_budget(1, volume.RSS_LIMIT)
        with self.assertRaisesRegex(ValueError, "WALL_BUDGET"):
            volume.enforce_budget(6600, 0)
        with self.assertRaisesRegex(ValueError, "RSS_BUDGET"):
            volume.enforce_budget(1, volume.RSS_LIMIT + 1)

    def test_coordinator_stops_sibling_without_refilling_failed_job(self):
        # Real child processes, no training. A failure must stop the other
        # active job before a new fold is launched.
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            args = SimpleNamespace(output=output, payload=OLD)
            pidfile = output / "sibling.pid"
            def child_command(_args, _role, *extra):
                if extra[-1] == "four120":
                    return [sys.executable, "-c", "import sys,time; time.sleep(.3); sys.exit(23)"]
                source = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(25)"
                return [sys.executable, "-c", source, str(pidfile)]
            with patch.object(volume, "preflight", return_value=(self.fmt, None, None, None, None, {})), \
                    patch.object(volume, "command", side_effect=child_command):
                with self.assertRaisesRegex(ValueError, "failed with code 23"):
                    volume.coordinate(args)
            self.assertEqual(sorted(p.name for p in (output / "jobs").iterdir()), ["f0-four120", "f0-four240"])
            self.assertFalse(volume.psutil.pid_exists(int(pidfile.read_text())))


if __name__ == "__main__":
    unittest.main()
