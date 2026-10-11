from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class UnsupportedCheckpointObject:
    pass


@unittest.skipIf(torch is None or np is None, "Torch/NumPy optional in CPU CI")
class CheckpointSafetyTests(unittest.TestCase):
    def test_legacy_numpy_rng_state_loads_without_unrestricted_pickle(self):
        from frontier350.secure_checkpoint350 import load_checkpoint
        payload = {
            "model": {"weight": torch.arange(5)},
            "numpy_rng": np.random.get_state(),
            "processed_tokens": 1_010_003_584,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(payload, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["processed_tokens"], 1_010_003_584)
            self.assertEqual(loaded["numpy_rng"][0], "MT19937")
            self.assertTrue(torch.equal(loaded["model"]["weight"], payload["model"]["weight"]))

    def test_arbitrary_pickle_object_is_rejected(self):
        from frontier350.secure_checkpoint350 import load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown.pt"
            torch.save({"model": {"w": torch.ones(1)},
                        "unexpected": UnsupportedCheckpointObject()}, path)
            with self.assertRaises(Exception):
                load_checkpoint(path)

    def test_requires_model_tensor_dictionary(self):
        from frontier350.secure_checkpoint350 import load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-a-model.pt"
            torch.save({"numpy_rng": np.random.get_state()}, path)
            with self.assertRaisesRegex(ValueError, "model tensor dictionary"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
