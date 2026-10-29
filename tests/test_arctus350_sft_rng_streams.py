"""Real sampler RNG regression checks; synthetic data only, no torch or GPU."""
import ast
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from frontier350.sft_batching350 import compact_length_batches, length_bucket_batches
from frontier350.sft_stratified350 import stratified_batches

ROOT = Path(__file__).resolve().parents[1]
ROWS = [SimpleNamespace(input_ids=list(range(i))) for i in range(5, 85)]


def replay_draws(mode, train_rng, replay_rng):
    if mode == "bucketed":
        batches = length_bucket_batches(ROWS, 4, train_rng, bucket_size=64)
    elif mode == "human-balanced":
        batches = stratified_batches(ROWS, [i % 2 == 0 for i in range(len(ROWS))],
                                     4, train_rng, human_per_batch=2, bucket_size=64)
    else:
        batches = compact_length_batches(ROWS, 4, train_rng)
    draws = []
    for step in range(1, 17):
        next(batches)
        if step % 4 == 0:
            # Same RNG calls and order as replay_batch for two synthetic shards.
            draws.extend((replay_rng.choice(("a", "b", "c")),
                          replay_rng.randrange(0, 10000)) for _ in range(2))
    return draws


class SftRngStreamsTests(unittest.TestCase):
    def test_existing_shared_rng_changes_replay_when_batching_changes(self):
        a, b = random.Random(351), random.Random(351)
        self.assertNotEqual(replay_draws("bucketed", a, a),
                            replay_draws("compact", b, b))

    def test_independent_replay_stream_is_same_across_samplers_and_repeatable(self):
        from frontier350.sft_rng350 import training_rngs
        a, r_a = training_rngs(350, "independent")
        b, r_b = training_rngs(350, "independent")
        c, r_c = training_rngs(350, "independent")
        d, r_d = training_rngs(350, "independent")
        first = replay_draws("bucketed", a, r_a)
        self.assertEqual(first, replay_draws("compact", b, r_b))
        self.assertEqual(first, replay_draws("human-balanced", d, r_d))
        self.assertEqual(first, replay_draws("bucketed", c, r_c))
        self.assertIsNot(a, r_a)

    def test_legacy_mode_reproduces_original_shared_rng_sequence(self):
        from frontier350.sft_rng350 import training_rngs
        for mode in ("bucketed", "compact", "human-balanced"):
            with self.subTest(mode=mode):
                original = random.Random(351)
                train_rng, replay_rng = training_rngs(350, "legacy")
                self.assertIs(train_rng, replay_rng)
                self.assertEqual(replay_draws(mode, original, original),
                                 replay_draws(mode, train_rng, replay_rng))

    def test_actual_replay_batch_reads_identical_synthetic_tokens(self):
        from frontier350.sft_rng350 import training_rngs

        class Tensor:
            def __init__(self, value):
                self.value = value
            def to(self, device, non_blocking=True):
                return self.value

        fake_torch = SimpleNamespace(
            from_numpy=lambda value: value,
            stack=lambda values: Tensor(np.stack(values)),
        )
        tree = ast.parse((ROOT / "frontier350/train_chat350.py").read_text())
        function = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "replay_batch")
        scope = {"np": np, "torch": fake_torch}
        compiled = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(compiled, "train_chat350.py", "exec"), scope)

        with tempfile.TemporaryDirectory() as folder:
            shards = []
            for index, length in enumerate((1000, 1200, 1400)):
                path = Path(folder) / f"{index}.bin"
                (np.arange(length, dtype=np.uint16) + index * 2000).tofile(path)
                shards.append(path)

            def actual_batches(mode):
                train_rng, replay_rng = training_rngs(350, "independent")
                batches = (length_bucket_batches(ROWS, 4, train_rng, bucket_size=64)
                           if mode == "bucketed" else compact_length_batches(ROWS, 4, train_rng))
                results = []
                for step in range(1, 17):
                    next(batches)
                    if step % 4 == 0:
                        x, y = scope["replay_batch"](shards, 2, 16, "cpu", replay_rng, 16000)
                        results.append((x.tolist(), y.tolist()))
                return results
            self.assertEqual(actual_batches("bucketed"), actual_batches("compact"))

    def test_invalid_seeds_and_stream_modes_fail_closed(self):
        from frontier350.sft_rng350 import training_rngs
        for seed, mode in ((True, "legacy"), (1.5, "legacy"), (2, "other")):
            with self.subTest(seed=seed, mode=mode), self.assertRaises(ValueError):
                training_rngs(seed, mode)

    def test_actual_training_loop_uses_the_streams_and_keeps_legacy_default(self):
        tree = ast.parse((ROOT / "frontier350/train_chat350.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        switches = [n for n in calls if n.args and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == "--rng-streams"]
        self.assertEqual(len(switches), 1)
        default = next(k.value for k in switches[0].keywords if k.arg == "default")
        self.assertEqual(ast.literal_eval(default), "legacy")
        init = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                and n.value.func.id == "training_rngs"]
        self.assertEqual(len(init), 1)
        self.assertEqual([n.id for n in init[0].targets[0].elts], ["rng", "replay_rng"])
        replay = [n for n in calls if isinstance(n.func, ast.Name)
                  and n.func.id == "replay_batch"]
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay[0].args[4].id, "replay_rng")
        for name, position in (("stratified_batches", 3),
                               ("compact_length_batches", 2),
                               ("length_bucket_batches", 2)):
            sampler = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == name]
            self.assertEqual(len(sampler), 1)
            self.assertEqual(sampler[0].args[position].id, "rng")


if __name__ == "__main__":
    unittest.main()
